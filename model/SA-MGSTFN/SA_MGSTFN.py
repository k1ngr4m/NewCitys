import importlib.util
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

_SEMANTIC_UTILS_PATH = os.path.join(os.path.dirname(__file__), "semantic_utils.py")
_SEMANTIC_SPEC = importlib.util.spec_from_file_location("sa_mgstfn_semantic_utils", _SEMANTIC_UTILS_PATH)
_SEMANTIC_UTILS = importlib.util.module_from_spec(_SEMANTIC_SPEC)
_SEMANTIC_SPEC.loader.exec_module(_SEMANTIC_UTILS)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        encoding = torch.zeros(max_len, embed_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[:encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0).unsqueeze(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoding[:, :x.size(1)].expand(x.size(0), -1, x.size(2), -1)


class TemporalContext(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.day_embedding = nn.Embedding(1441, embed_dim)
        self.week_embedding = nn.Embedding(8, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, output_dim: int) -> torch.Tensor:
        if x.size(-1) <= output_dim + 1:
            return torch.zeros(x.size(0), x.size(1), x.size(2), self.day_embedding.embedding_dim, device=x.device)
        day = x[..., output_dim].long().clamp(0, 1440)
        week = x[..., output_dim + 1].long().clamp(0, 7)
        return self.proj(torch.cat([self.day_embedding(day), self.week_embedding(week)], dim=-1))


class GraphAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float, drop: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_nodes, _ = x.shape
        flat_x = x.reshape(batch_size * time_steps, num_nodes, self.embed_dim)
        query = self.q_proj(flat_x).view(batch_size * time_steps, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(flat_x).view(batch_size * time_steps, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(flat_x).view(batch_size * time_steps, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        mask = adjacency.to(dtype=torch.bool, device=x.device).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~mask, -1e9)
        attention = self.attn_drop(torch.softmax(scores, dim=-1))
        out = torch.matmul(attention, value).transpose(1, 2).reshape(batch_size * time_steps, num_nodes, self.embed_dim)
        out = self.drop(self.out_proj(out))
        return out.view(batch_size, time_steps, num_nodes, self.embed_dim)


class TimeAwareMultiGraphFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float, drop: float, mlp_ratio: int):
        super().__init__()
        self.phy_gat = GraphAttention(embed_dim, num_heads, attn_drop, drop)
        self.flow_gat = GraphAttention(embed_dim, num_heads, attn_drop, drop)
        self.sem_gat = GraphAttention(embed_dim, num_heads, attn_drop, drop)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )
        hidden_dim = embed_dim * mlp_ratio
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(drop),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_context: torch.Tensor,
        physical_adj: torch.Tensor,
        flow_adj: torch.Tensor,
        semantic_adj: torch.Tensor,
    ) -> torch.Tensor:
        norm_x = self.norm1(x)
        phy = self.phy_gat(norm_x, physical_adj)
        flow = self.flow_gat(norm_x, flow_adj)
        sem = self.sem_gat(norm_x, semantic_adj)
        gate_input = torch.cat([norm_x, time_context], dim=-1)
        weights = torch.softmax(self.fusion_gate(gate_input), dim=-1).unsqueeze(-1)
        fused = weights[..., 0, :] * phy + weights[..., 1, :] * flow + weights[..., 2, :] * sem
        x = x + fused
        return x + self.ffn(self.norm2(x))


class DecompositionRefiner(nn.Module):
    def __init__(self, embed_dim: int, drop: float):
        super().__init__()
        self.inv_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim, embed_dim),
        )
        self.spe_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim, embed_dim),
        )
        self.reconstruct = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x: torch.Tensor, time_context: torch.Tensor):
        z_inv = self.inv_encoder(torch.cat([x, time_context], dim=-1))
        z_spe = self.spe_encoder(x)
        combined = torch.cat([z_inv, z_spe], dim=-1)
        reconstructed = self.reconstruct(combined)
        return z_inv, z_spe, combined, reconstructed


class SAMGSTFN(nn.Module):
    def __init__(self, args, dataset_use, device, dim_in: int, dim_out: int):
        super().__init__()
        self.device = device
        self.dataset_use = dataset_use
        self.output_dim = dim_out
        self.input_base_dim = dim_in
        self.input_window = args.input_window
        self.output_window = args.output_window
        self.embed_dim = args.embed_dim
        self.semantic_dim = args.semantic_dim
        self.semantic_threshold = args.semantic_threshold
        self.flow_topk = args.flow_topk
        self.lambda_orth = args.lambda_orth
        self.lambda_recon = args.lambda_recon
        self.orth_loss = torch.tensor(0.0)
        self.recon_loss = torch.tensor(0.0)
        self.auxiliary_loss = torch.tensor(0.0)

        self.adj_mx_dict = args.adj_mx_dict
        self.raw_adj_mx_dict = args.raw_adj_mx_dict
        self.semantic_embedding_dict = nn.ParameterDict()

        self.input_proj = nn.Linear(dim_in, self.embed_dim)
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.temporal_context = TemporalContext(self.embed_dim)
        self.semantic_domain_mapper = nn.Sequential(
            nn.Linear(args.semantic_raw_dim, self.semantic_dim),
            nn.ReLU(),
            nn.Linear(self.semantic_dim, self.embed_dim),
            nn.ReLU(),
        )
        self.semantic_feature_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.encoder_layers = nn.ModuleList([
            TimeAwareMultiGraphFusion(
                embed_dim=self.embed_dim,
                num_heads=args.num_heads,
                attn_drop=args.attn_drop,
                drop=args.drop,
                mlp_ratio=args.mlp_ratio,
            )
            for _ in range(args.num_layers)
        ])
        self.refiner = DecompositionRefiner(self.embed_dim, args.drop)
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Dropout(args.drop),
            nn.Linear(self.embed_dim, self.output_dim),
        )
        self.horizon_proj = nn.Linear(self.input_window, self.output_window)

        self._load_semantic_embeddings(args)

    def _load_semantic_embeddings(self, args):
        for dataset in self.dataset_use:
            texts = _SEMANTIC_UTILS.build_or_load_semantic_texts(
                dataset=dataset,
                meta_path=args.semantic_meta_path_dict[dataset],
                cache_dir=args.semantic_cache_dir_dict[dataset],
                radius=args.osm_radius,
                endpoint=args.osm_endpoint,
                timeout=args.osm_timeout,
                force_refresh=args.semantic_force_refresh,
                disable_osm=args.semantic_disable_osm,
            )
            embeddings = _SEMANTIC_UTILS.build_or_load_distilbert_embeddings(
                texts=texts,
                cache_dir=args.semantic_cache_dir_dict[dataset],
                model_name=args.semantic_model_name,
                batch_size=args.semantic_batch_size,
                max_length=args.semantic_max_length,
                device=self.device,
                force_refresh=args.semantic_force_refresh,
            )
            if embeddings.shape[1] != args.semantic_raw_dim:
                raise ValueError(
                    f"Expected DistilBERT embeddings with dim {args.semantic_raw_dim}, got {embeddings.shape[1]}."
                )
            self.semantic_embedding_dict[dataset] = nn.Parameter(torch.FloatTensor(embeddings), requires_grad=False)

    @staticmethod
    def _normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency.float()
        degree = adjacency.sum(dim=-1).clamp_min(1e-6)
        degree_inv_sqrt = torch.rsqrt(degree)
        return degree_inv_sqrt.unsqueeze(-1) * adjacency * degree_inv_sqrt.unsqueeze(0)

    def _semantic_adjacency(self, select_dataset: str) -> torch.Tensor:
        semantic_raw = self.semantic_embedding_dict[select_dataset].to(self.device)
        semantic_features = self.semantic_domain_mapper(semantic_raw)
        semantic_features = F.normalize(semantic_features, p=2, dim=-1)
        similarity = torch.matmul(semantic_features, semantic_features.transpose(0, 1)).clamp_min(0.0)
        semantic_adj = torch.where(similarity >= self.semantic_threshold, similarity, torch.zeros_like(similarity))
        semantic_adj = semantic_adj + torch.eye(semantic_adj.size(0), device=semantic_adj.device)
        return self._normalize_adjacency(semantic_adj)

    def _flow_adjacency(self, source: torch.Tensor) -> torch.Tensor:
        traffic = source[..., :self.input_base_dim].mean(dim=(0, -1)).transpose(0, 1)
        traffic = traffic - traffic.mean(dim=-1, keepdim=True)
        traffic = traffic / traffic.std(dim=-1, keepdim=True).clamp_min(1e-6)
        correlation = torch.matmul(traffic, traffic.transpose(0, 1)) / max(traffic.size(-1), 1)
        correlation = correlation.clamp_min(0.0)
        if self.flow_topk > 0 and self.flow_topk < correlation.size(-1):
            topk_values, topk_indices = torch.topk(correlation, k=self.flow_topk, dim=-1)
            sparse = torch.zeros_like(correlation)
            sparse.scatter_(dim=-1, index=topk_indices, src=topk_values)
            correlation = sparse
        correlation = correlation + torch.eye(correlation.size(0), device=correlation.device)
        return self._normalize_adjacency(correlation)

    def _compute_auxiliary_losses(self, z_inv: torch.Tensor, z_spe: torch.Tensor, reconstructed: torch.Tensor, encoded: torch.Tensor):
        inv_flat = F.normalize(z_inv.reshape(-1, z_inv.size(-1)), p=2, dim=-1)
        spe_flat = F.normalize(z_spe.reshape(-1, z_spe.size(-1)), p=2, dim=-1)
        self.orth_loss = torch.abs(torch.sum(inv_flat * spe_flat, dim=-1)).mean()
        self.recon_loss = F.mse_loss(reconstructed, encoded)
        self.auxiliary_loss = self.lambda_orth * self.orth_loss + self.lambda_recon * self.recon_loss

    def forward(self, source: torch.Tensor, select_dataset: str) -> torch.Tensor:
        x = source[..., :self.input_base_dim]
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized = centered / stdev

        encoded = self.input_proj(normalized) + self.position_encoding(normalized)
        time_context = self.temporal_context(source, self.output_dim)
        semantic_raw = self.semantic_embedding_dict[select_dataset].to(self.device)
        semantic_context = self.semantic_feature_proj(self.semantic_domain_mapper(semantic_raw)).unsqueeze(0).unsqueeze(0)
        encoded = encoded + semantic_context

        physical_adj = self.adj_mx_dict[select_dataset].to(self.device)
        flow_adj = self._flow_adjacency(source)
        semantic_adj = self._semantic_adjacency(select_dataset)

        for layer in self.encoder_layers:
            encoded = layer(encoded, time_context, physical_adj, flow_adj, semantic_adj)

        z_inv, z_spe, combined, reconstructed = self.refiner(encoded, time_context)
        self._compute_auxiliary_losses(z_inv, z_spe, reconstructed, encoded)

        decoded = self.decoder(combined)
        decoded = self.horizon_proj(decoded.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        decoded = decoded * stdev[:, :1] + means[:, :1]
        return decoded
