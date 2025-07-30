import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from peft import get_peft_model, LoraConfig
from transformers import GPT2Model


@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(2)].unsqueeze(1).expand_as(x)

class PatchEmbedding_flow(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, his):
        super(PatchEmbedding_flow, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.his = his

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        # do patching
        x = x.squeeze(-1).permute(0, 2, 1)
        if self.his == x.shape[-1]:
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        else:
            gap = self.his // x.shape[-1]
            x = x.unfold(dimension=-1, size=self.patch_len//gap, step=self.stride//gap)
            x = F.pad(x, (0, (self.patch_len - self.patch_len//gap)))
        x = self.value_embedding(x)
        x = x + self.position_encoding(x)
        x = x.permute(0, 2, 1, 3)
        return x

class PatchEmbedding_time(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, his):
        super(PatchEmbedding_time, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.his = his
        self.minute_size = 1440 + 1
        self.daytime_embedding = nn.Embedding(self.minute_size, d_model//2)
        weekday_size = 7 + 1
        self.weekday_embedding = nn.Embedding(weekday_size, d_model//2)

    def forward(self, x):
        # do patching
        bs, ts, nn, dim = x.size()
        x = x.permute(0, 2, 3, 1).reshape(bs, -1, ts)
        if self.his == x.shape[-1]:
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        else:
            gap = self.his // x.shape[-1]
            x = x.unfold(dimension=-1, size=self.patch_len//gap, step=self.stride//gap)
        num_patch = x.shape[-2]
        x = x.reshape(bs, nn, dim, num_patch, -1).transpose(1, 3)
        x_tdh = x[:, :, 0, :, 0]
        x_dwh = x[:, :, 1, :, 0]
        x_tdp = x[:, :, 2, :, 0]
        x_dwp = x[:, :, 3, :, 0]

        x_tdh = self.daytime_embedding(x_tdh)
        x_dwh = self.weekday_embedding(x_dwh)
        x_tdp = self.daytime_embedding(x_tdp)
        x_dwp = self.weekday_embedding(x_dwp)
        x_th = torch.cat([x_tdh, x_dwh], dim=-1)
        x_tp = torch.cat([x_tdp, x_dwp], dim=-1)

        return x_th, x_tp

class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, prob_drop, alpha):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=False)
        self.mlp = nn.Linear(out_dim, out_dim)
        self.dropout = prob_drop
        self.alpha = alpha

    def forward(self, x, adj):
        d = adj.sum(1)
        h = x
        a = adj / d.view(-1, 1)
        gcn_out = self.fc1(torch.einsum('bdkt,nk->bdnt', h, a))
        out = self.alpha*x + (1-self.alpha)*gcn_out
        ho = self.mlp(out)
        return ho

class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, prob_drop, alpha, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.alpha = alpha
        self.head_dim = out_dim // num_heads
        assert self.head_dim * num_heads == out_dim, "头数须能整除输出维度"

        # 投影矩阵
        self.W = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)

        # 注意力参数
        self.a_src = nn.Parameter(torch.Tensor(num_heads, self.head_dim, 1))
        self.a_dst = nn.Parameter(torch.Tensor(num_heads, self.head_dim, 1))

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(prob_drop)
        self.mlp = nn.Linear(out_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.W.weight)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, x, adj):
        B, T, N, D_in = x.shape
        x_flat = x.reshape(B * T, N, D_in)

        # 投影到多头空间 [B*T, N, nh, hd]
        h = self.W(x_flat).view(B * T, N, self.num_heads, self.head_dim)
        h = h.transpose(1, 2)  # [B*T, nh, N, hd]

        # 计算注意力得分（广播机制）
        e_src = torch.matmul(h, self.a_src)  # [B*T, nh, N, 1]
        e_dst = torch.matmul(h, self.a_dst)  # [B*T, nh, N, 1]
        e = e_src + e_dst.transpose(-1, -2)  # [B*T, nh, N, N]
        e = self.leakyrelu(e)

        # 应用邻接矩阵mask
        adj_expanded = adj.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        e = e.masked_fill(adj_expanded == 0, -1e9)

        # 计算注意力权重
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)

        # 聚合特征
        out = torch.matmul(attention, h)  # [B*T, nh, N, hd]
        out = out.transpose(1, 2).reshape(B * T, N, -1)  # [B*T, N, out_dim]
        out = out.view(B, T, N, -1)  # 恢复形状

        # 残差连接 + MLP
        gat_out = self.alpha * x + (1 - self.alpha) * out
        return self.mlp(gat_out)

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()

        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb

class PFA(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6, U=1, dropout_rate=0.0, gpt_model_path="llm-model/gpt2"):
        super(PFA, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(gpt_model_path, attn_implementation="eager",
                                              output_attentions=True, output_hidden_states=True)
        
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U
        self.device = device
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.lora_rank = 16

        self.lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=32,
            lora_dropout=self.dropout_rate,
            target_modules=['q_attn','c_attn'],
            bias="none"
        )

        self.gpt2 = get_peft_model(self.gpt2, self.lora_config)

        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def custom_forward(self,
                    input_ids: Optional[torch.LongTensor] = None,
                    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                    attention_mask: Optional[torch.FloatTensor] = None,
                    token_type_ids: Optional[torch.LongTensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    head_mask: Optional[torch.FloatTensor] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    encoder_hidden_states: Optional[torch.Tensor] = None,
                    encoder_attention_mask: Optional[torch.FloatTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None,
                    adjacency_matrix: Optional[torch.FloatTensor] = None, 
                    ) -> Union[Tuple, dict]:

        output_attentions = output_attentions if output_attentions is not None else self.gpt2.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.gpt2.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.gpt2.config.use_cache
        return_dict = return_dict if return_dict is not None else self.gpt2.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.gpt2.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.gpt2.wte(input_ids)
        position_embeds = self.gpt2.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        presents = () if use_cache else None

        total_layers = len(self.gpt2.h)

        for i, (block, layer_past) in enumerate(zip(self.gpt2.h, past_key_values)):
            if i >= total_layers - self.U and adjacency_matrix is not None:
                attention_mask = adjacency_matrix.to(hidden_states.device).float()
            elif attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]

            if use_cache:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
        
        hidden_states = self.gpt2.ln_f(hidden_states)
        hidden_states = hidden_states.view((-1,) + input_shape[1:] + (hidden_states.size(-1),))

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )

    def forward(self, x, adjacency_matrix):
        """
        Args:
            x: input embeddings [batch_size, sequence_length, hidden_dim]
            adjacency_matrix: adjacency matrix used as an attention mask
                              [batch_size, sequence_length, sequence_length]
        """
        batch_size =  x.shape[0]
        num_heads =  self.gpt2.config.n_head
        adjacency_matrix = adjacency_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
        adjacency_matrix = adjacency_matrix.unsqueeze(1).repeat(1, num_heads, 1, 1)

        attention_mask = adjacency_matrix.to(self.device).float()

        # Use GPT-2 with attention mask
        output = self.custom_forward(
            inputs_embeds=x,
            attention_mask=attention_mask
        ).last_hidden_state
        output = self.dropout(output)

        return output

class TemporalSelfAttention(nn.Module):
    def __init__(
        self, dim, t_attn_size, t_num_heads=6, tc_num_heads=6, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'),
    ):
        super().__init__()
        assert dim % t_num_heads == 0
        self.t_num_heads = t_num_heads
        self.tc_num_heads = tc_num_heads
        self.head_dim = dim // t_num_heads
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.t_attn_size = t_attn_size

        self.t_q_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.t_k_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.t_v_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.norm_tatt1 = LlamaRMSNorm(dim)
        self.norm_tatt2 = LlamaRMSNorm(dim)

        self.tc_q_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.tc_k_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.tc_v_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.tc_attn_drop = nn.Dropout(attn_drop)

        self.GCN = GCN(dim, dim, proj_drop, alpha=0.05)
        self.GAT = GAT(dim, dim, proj_drop, alpha=0.05, num_heads=4)
        self.act = nn.GELU()

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_k, x_v, TH, TP, adj, geo_mask=None, sem_mask=None, trg_mask=False):
        B, T_q, N, D = x_q.shape
        T_k, T_v = x_k.shape[1], x_v.shape[1]

        tc_q = self.tc_q_conv(TP).transpose(1, 2)
        tc_k = self.tc_k_conv(TH).transpose(1, 2)
        tc_v = self.tc_v_conv(x_q).transpose(1, 2)
        tc_q = tc_q.reshape(B, N, T_q, self.tc_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        tc_k = tc_k.reshape(B, N, T_k, self.tc_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        tc_v = tc_v.reshape(B, N, T_v, self.tc_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        tc_attn = (tc_q @ tc_k.transpose(-2, -1)) * self.scale
        if trg_mask:
            ones = torch.ones_like(tc_attn).to(self.device)
            dec_mask = torch.triu(ones, diagonal=1)
            tc_attn = tc_attn.masked_fill(dec_mask == 1, -1e9)
        tc_attn = tc_attn.softmax(dim=-1)
        tc_attn = self.tc_attn_drop(tc_attn)
        tc_x = (tc_attn @ tc_v).transpose(2, 3).reshape(B, N, T_q, D).transpose(1, 2)

        tc_x = self.norm_tatt1(tc_x + x_q)

        t_q = self.t_q_conv(tc_x).transpose(1, 2)
        t_k = self.t_k_conv(tc_x).transpose(1, 2)
        t_v = self.t_v_conv(tc_x).transpose(1, 2)
        t_q = t_q.reshape(B, N, T_q, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T_k, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T_v, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        if trg_mask:
            ones = torch.ones_like(t_attn).to(self.device)
            dec_mask = torch.triu(ones, diagonal=1)
            t_attn = t_attn.masked_fill(dec_mask == 1, -1e9)
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T_q, D).transpose(1, 2)

        t_x = self.norm_tatt2(t_x + tc_x)
        gcn_out = self.GCN(t_x, adj)
        gat_out = self.GAT(t_x, adj)
        x = self.proj_drop(gat_out+gcn_out)
        return x

class STEncoderBlock(nn.Module):
    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=4, tc_num_heads=4, t_num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, device=torch.device('cpu'), type_ln="pre", output_dim=1,
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = LlamaRMSNorm(dim)
        self.norm2 = LlamaRMSNorm(dim)
        self.st_attn = TemporalSelfAttention(dim, t_attn_size, t_num_heads=t_num_heads, tc_num_heads=tc_num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, device=device)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(hidden_size=dim, intermediate_size=mlp_hidden_dim)

    def forward(self, x, dec_in, enc_out, TH, TP, adj, geo_mask=None, sem_mask=None):
        if self.type_ln == 'pre':
            x_nor1 = self.norm1(x)
            x = x + self.drop_path(self.st_attn(x_nor1, x_nor1, x_nor1, TH, TP, adj, geo_mask=geo_mask, sem_mask=sem_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            x = self.norm1((x + self.drop_path(self.st_attn(x, x, x, TH, TP, adj, geo_mask=geo_mask, sem_mask=sem_mask))))
            x = self.norm2((x + self.drop_path(self.mlp(x))))
        else:
            x = x + self.drop_path(self.st_attn(x, x, x, TH, TP, adj, geo_mask=geo_mask, sem_mask=sem_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class NewCityPlus(nn.Module):
    def __init__(self, args, dataset_use, device, dim_in):
        super(NewCityPlus, self).__init__()

        self.feature_dim = dim_in
        self.adj_mx_dict = args.adj_mx_dict
        self.sh_mx_dict = args.sh_mx_dict
        self.lap_mx_dict = args.lap_mx_dict

        self.embed_dim = args.embed_dim
        self.skip_dim = args.skip_dim
        self.lape_dim = args.lape_dim
        self.geo_num_heads = args.geo_num_heads
        self.sem_num_heads = args.sem_num_heads
        self.tc_num_heads = args.tc_num_heads
        self.t_num_heads = args.t_num_heads
        self.mlp_ratio = args.mlp_ratio
        self.qkv_bias = args.qkv_bias
        self.drop = args.drop
        self.attn_drop = args.attn_drop
        self.drop_path = args.drop_path
        self.s_attn_size = args.s_attn_size
        self.t_attn_size = args.t_attn_size
        self.enc_depth = args.enc_depth
        self.type_ln = args.type_ln
        self.llm_layer = getattr(args, 'llm_layer', 6)  # ST-LLM-Plus特性
        self.U = getattr(args, 'U', 1)  # ST-LLM-Plus特性

        self.output_dim = dim_in
        self.input_window = args.input_window
        self.output_window = args.output_window
        self.device = device
        self.far_mask_delta = args.far_mask_delta
        self.weather_dim = args.weather_dim
        print(f"weather_dim:{self.weather_dim}")

        self.geo_mask_dict = {}
        for i, data_graph in enumerate(dataset_use):
            sh_mx = self.sh_mx_dict[data_graph].T
            self.geo_mask_dict[data_graph] = torch.zeros_like(sh_mx)
            self.geo_mask_dict[data_graph][sh_mx >= self.far_mask_delta] = 1
            self.geo_mask_dict[data_graph] = self.geo_mask_dict[data_graph].bool()
        self.sem_mask = None

        self.patch_embedding_flow = PatchEmbedding_flow(
            self.embed_dim, patch_len=12, stride=12, padding=0, his=args.input_window)
        self.patch_embedding_time = PatchEmbedding_time(
            self.embed_dim, patch_len=12, stride=12, padding=0, his=args.input_window)
        self.spatial_embedding = LaplacianPE(self.lape_dim, self.embed_dim)
        
        # ST-LLM-Plus特性：节点嵌入
        # 从邻接矩阵获取节点数
        nyc_nodes = self.adj_mx_dict.get('NYC_TAXI', torch.eye(263)).shape[0]
        self.node_emb = nn.Parameter(torch.empty(nyc_nodes, self.embed_dim))
        nn.init.xavier_uniform_(self.node_emb)
        
        # ST-LLM-Plus特性：时间嵌入
        self.Temb = TemporalEmbedding(288, self.embed_dim)

        enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim, s_attn_size=self.s_attn_size, t_attn_size=self.t_attn_size,
                geo_num_heads=self.geo_num_heads, sem_num_heads=self.sem_num_heads, tc_num_heads=self.tc_num_heads, t_num_heads=self.t_num_heads,
                mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop, attn_drop=self.attn_drop, drop_path=enc_dpr[i], act_layer=nn.GELU,
                device=self.device, type_ln=self.type_ln, output_dim=self.output_dim,
            ) for i in range(self.enc_depth)
        ])

        # ST-LLM-Plus特性：GPT2模型
        self.gpt = PFA(device=self.device, gpt_layers=self.llm_layer, U=self.U, dropout_rate=self.drop, gpt_model_path=getattr(args, 'gpt_model_path', "llm-model/gpt2"))
        
        # ST-LLM-Plus特性：输入层和回归层
        self.in_layer = nn.Conv2d(self.embed_dim*3, 768, kernel_size=(1, 1))  # 3个嵌入：流量、时间、节点
        self.regression_layer = nn.Conv2d(768, self.output_window, kernel_size=(1, 1))

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(24*self.skip_dim, self.output_window)
        self.weather_fc = nn.Linear(self.weather_dim * self.patch_embedding_flow.patch_len, args.embed_dim)

    def forward(self, input, lbls, select_dataset):
        bs, time_steps, num_nodes, num_feas = input.size()
        x = input
        
        # Spatio-Temporal Context Encoding
        TCH = input[..., self.output_dim:].long()
        TCP = lbls[..., self.output_dim:].long()
        feas_all_his, feas_all_pre = self.patch_embedding_time(torch.cat([TCH, TCP], dim=-1))
        spa_feas = self.spatial_embedding(self.lap_mx_dict[select_dataset].to(self.device)).repeat(bs, feas_all_his.shape[1], 1, 1)
        feas_all_his = feas_all_his + spa_feas
        feas_all_pre = feas_all_pre + spa_feas

        # IN
        x_in = x[..., :self.output_dim]
        means = x_in.mean(1, keepdim=True).detach()
        x_in = x_in - means
        stdev = torch.sqrt(torch.var(x_in, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach()
        x_in /= stdev

        # Patch Embedding
        enc = self.patch_embedding_flow(x_in)

        # 假设天气数据在输入数据的最后 weather_dim 个维度
        weather_data = input[..., -self.weather_dim:]

        if self.weather_dim > 0:
            # 调整维度顺序并分片
            B, T, N, W = weather_data.size()
            # 转换为 [B, N, W, T] 并合并批次和节点
            weather_data = weather_data.permute(0, 2, 3, 1).reshape(B * N, W, T)

            # 分片参数
            patch_len = self.patch_embedding_flow.patch_len
            stride = self.patch_embedding_flow.stride

            # 在时间维度上进行分片
            weather_patched = weather_data.unfold(
                dimension=-1,
                size=patch_len,
                step=stride
            )  # [B*N, W, num_patches, patch_len]

            num_patches = weather_patched.shape[-2]
            # 调整形状并分离批次和节点
            weather_patched = weather_patched.reshape(B, N, W, num_patches, patch_len)
            # 合并天气维度和分片长度，并调整维度顺序
            weather_patched = weather_patched.permute(0, 3, 1, 2, 4)  # [B, num_patches, N, W, patch_len]
            weather_patched = weather_patched.reshape(B * num_patches * N, -1)  # 展平最后三个维度

            # 投影到嵌入空间
            weather_embedding = self.weather_fc(weather_patched)
            # 调整为与enc匹配的形状 [B, num_patches, N, embed_dim]
            weather_embedding = weather_embedding.reshape(B, num_patches, N, -1)

            enc = enc + weather_embedding
            
        # ST-LLM-Plus特性：添加节点嵌入和时间嵌入
        node_emb = []
        node_emb.append(
            self.node_emb.unsqueeze(0)
            .expand(bs, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )
        
        # 时间嵌入
        tem_emb = self.Temb(input.permute(0, 3, 2, 1))
        
        # 合并所有嵌入
        data_st = torch.cat([enc.permute(0, 3, 2, 1)] + [tem_emb] + node_emb, dim=1)
        data_st = self.in_layer(data_st)
        data_st = F.leaky_relu(data_st)
        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)
        
        # 使用GPT2处理
        adj = self.adj_mx_dict[select_dataset].to(self.device)
        outputs = self.gpt(data_st, adj)
        
        # 回归层
        outputs = outputs.permute(0, 2, 1).unsqueeze(-1)
        outputs = self.regression_layer(outputs)

        # 保持与NewCity相同的输出格式
        skip = outputs.permute(0, 3, 2, 1)
        skip = skip[:, :time_steps, :, :]

        # DeIN
        skip = skip * stdev
        skip = skip + means

        return skip