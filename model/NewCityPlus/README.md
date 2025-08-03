# NewCityPlus 模型

## 概述

NewCityPlus 是一个先进的时空交通预测模型，它在 NewCity 架构的基础上集成了大语言模型（LLM）。该模型旨在通过结合图神经网络和 GPT-2 参数高效微调技术来预测交通模式。

## 架构

### 核心组件

1. **Patch 嵌入**
   - 通过 patching 机制处理时间序列数据
   - 处理流量数据和时间特征

2. **时空注意力机制**
   - 专门用于空间和时间依赖关系的注意力机制
   - 带有时间上下文（TC）和时间（T）注意力头的 TemporalSelfAttention

3. **GPT-2 集成**
   - 使用 LoRA 的参数高效微调（PFA）方法
   - 选择性层冻结以实现高效训练

4. **图神经网络**
   - 结合 GCN 和 GAT 进行空间特征提取
   - 将邻接矩阵集成作为注意力掩码

5. **位置编码**
   - 拉普拉斯位置编码用于空间结构信息

### 关键特性

- **STEncoderBlock**: 处理时空特征的堆叠编码器块
- **多模态嵌入**: 流量 patch、时间上下文、拉普拉斯位置、节点和时间嵌入
- **自定义 GPT-2 前向传递**: 将图结构融入 LLM 处理过程

## 配置

### 关键参数

- `embed_dim`: 嵌入维度（基础为 128，NYC_TAXI 为 256）
- `input_window/output_window`: 序列长度（通常为 288 个时间步）
- `lape_dim`: 拉普拉斯位置编码维度
- `llm_layer`: 使用的 GPT-2 层数（6 层）
- `U`: GPT-2 中未冻结的层数（根据数据集为 1 或 2）

### 配置文件

1. `conf/NewCityPlus/NewCityPlus.conf`: 基础配置
2. `conf/NewCityPlus/NewCityPlus_NYC_TAXI.conf`: NYC_TAXI 特定的优化配置

## 使用方法

### 模型集成

模型通过 `model/Model.py` 集成：

```python
elif self.model == 'NewCityPlus':
    from model.NewCityPlus.NewCityPlus import NewCityPlus
    self.predictor = NewCityPlus(args_predictor, args.dataset_use, args.device, dim_in)
```

### 数据格式

输入数据结构：
- 形状: `[batch_size, time_steps, num_nodes, features]`
- 特征包括交通流量和时间信息
- 用于空间关系的邻接矩阵

### 训练过程

1. 数据预处理和归一化
2. 使用输入/输出窗口创建序列
3. 使用掩码 MAE 计算损失
4. 使用 Adam 优化器进行反向传播

## 关键创新点

### LLM 集成
- 使用 LoRA 的参数高效微调
- 选择性层冻结（仅顶层 U 层可训练）
- 将邻接矩阵集成作为注意力掩码

### 多模态嵌入
结合多种嵌入类型以实现全面的特征表示。





● NewCityPlus 模型详细介绍

  概述

  NewCityPlus 是一个先进的时空交通预测模型，它在 NewCity 架构的基础上集成了大语言模型（LLM），通过参数高效微调技术来有效预测交通模式。该模型将图神经网络与 GPT-2 相结合，实现了更准确的交通流量预测。       

  架构组件

  1. 主模型实现 (NewCityPlus.py)

  主实现文件位于 model/NewCityPlus/NewCityPlus.py。

  核心模块:

  Patch 嵌入组件:
  - PatchEmbedding_flow: 通过 patching 机制处理时间序列交通流量数据，包含值嵌入和位置编码
  - PatchEmbedding_time: 处理时间特征，包含一天中的时间和一周中的天数嵌入

  位置编码:
  - LaplacianPE: 实现拉普拉斯位置编码，用于捕获空间结构信息
  - PositionalEncoding: 标准正弦位置编码，用于时间序列

  图神经网络:
  - GCN: 图卷积网络，集成邻接矩阵
  - GAT: 图注意力网络，采用多头注意力机制

  注意力机制:
  - TemporalSelfAttention: 专门用于时间依赖性的注意力机制，包含时间上下文(TC)和时间(T)注意力头
  - STEncoderBlock: 堆叠的编码器块，处理时空特征

  LLM 集成:
  - PFA (参数高效微调适配器): 通过 LoRA (低秩适配) 和选择性层冻结来集成 GPT-2

  2. GPT-2 集成 (PFA - 参数高效微调适配器)

  PFA 模块实现了核心的 LLM 集成:

  class PFA(nn.Module):
      def __init__(self, device="cuda:0", gpt_layers=6, U=1, dropout_rate=0.0, gpt_model_path="llm/gpt2"):
          super(PFA, self).__init__()
          # 加载具有特定配置的 GPT-2 模型
          self.gpt2 = GPT2Model.from_pretrained(gpt_model_path, attn_implementation="eager",
                                                output_attentions=True, output_hidden_states=True)
          self.gpt2.h = self.gpt2.h[:gpt_layers]  # 仅使用指定层数
          self.U = U  # 未冻结的层数
          self.lora_config = LoraConfig(
              r=self.lora_rank,  # LoRA 秩
              lora_alpha=32,
              lora_dropout=self.dropout_rate,
              target_modules=['q_attn','c_attn'],  # 对特定模块应用 LoRA
              bias="none"
          )
          self.gpt2 = get_peft_model(self.gpt2, self.lora_config)  # 应用 LoRA 配置

          # 选择性层冻结 - 仅顶层 U 层可训练
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

  3. 时空注意力机制

  模型实现了专门的注意力机制来捕获空间和时间依赖性:

  class TemporalSelfAttention(nn.Module):
      def __init__(
          self, dim, t_attn_size, t_num_heads=6, tc_num_heads=6, qkv_bias=False,
          attn_drop=0., proj_drop=0., device=torch.device('cpu'),
      ):
          # 实现双重注意力 - 时间上下文(tc)和时间(t)注意力
          self.tc_q_conv = nn.Linear(dim, dim, bias=qkv_bias)  # 时间上下文查询
          self.tc_k_conv = nn.Linear(dim, dim, bias=qkv_bias)  # 时间上下文键
          self.tc_v_conv = nn.Linear(dim, dim, bias=qkv_bias)  # 时间上下文值
          self.t_q_conv = nn.Linear(dim, dim, bias=qkv_bias)   # 时间查询
          self.t_k_conv = nn.Linear(dim, dim, bias=qkv_bias)   # 时间键
          self.t_v_conv = nn.Linear(dim, dim, bias=qkv_bias)   # 时间值

  4. 拉普拉斯位置编码

  拉普拉斯位置编码捕获空间结构信息:

  class LaplacianPE(nn.Module):
      def __init__(self, lape_dim, embed_dim):
          super().__init__()
          self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

      def forward(self, lap_mx):
          lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
          return lap_pos_enc

  5. 图神经网络组件

  GCN (图卷积网络):
  class GCN(nn.Module):
      def __init__(self, in_dim, out_dim, prob_drop, alpha):
          super(GCN, self).__init__()
          self.fc1 = nn.Linear(in_dim, out_dim, bias=False)
          self.mlp = nn.Linear(out_dim, out_dim)
          self.alpha = alpha  # 残差连接权重

      def forward(self, x, adj):
          d = adj.sum(1)
          h = x
          a = adj / d.view(-1, 1)  # 归一化邻接矩阵
          gcn_out = self.fc1(torch.einsum('bdkt,nk->bdnt', h, a))
          out = self.alpha*x + (1-self.alpha)*gcn_out  # 残差连接
          return self.mlp(out)

  GAT (图注意力网络):
  class GAT(nn.Module):
      def __init__(self, in_dim, out_dim, prob_drop, alpha, num_heads=4):
          super().__init__()
          self.num_heads = num_heads
          self.W = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)
          # 注意力参数
          self.a_src = nn.Parameter(torch.Tensor(num_heads, self.head_dim, 1))
          self.a_dst = nn.Parameter(torch.Tensor(num_heads, self.head_dim, 1))

  6. 配置文件

  基础配置 (conf/NewCityPlus/NewCityPlus.conf):
  [model]
  embed_dim = 128
  skip_dim = 128
  lape_dim = 8
  tc_num_heads = 16
  t_num_heads = 16
  enc_depth = 3
  llm_layer = 6
  U = 1

  NYC_TAXI 优化配置 (conf/NewCityPlus/NewCityPlus_NYC_TAXI.conf):
  [model]
  embed_dim = 256
  skip_dim = 256
  enc_depth = 4
  U = 2

  7. 模型中的数据流

  1. 输入处理:
    - 输入形状: [batch_size, time_steps, num_nodes, features]
    - 交通流量数据和时间信息被分离
  2. 归一化:
  means = x_in.mean(1, keepdim=True).detach()
  x_in = x_in - means
  stdev = torch.sqrt(torch.var(x_in, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach()
  x_in /= stdev
  3. Patch 嵌入:
    - 流量数据通过 PatchEmbedding_flow 处理
    - 时间特征通过 PatchEmbedding_time 处理
  4. 多模态嵌入:
    - 流量 patch
    - 时间上下文嵌入
    - 拉普拉斯位置编码
    - 节点嵌入
    - 时间嵌入
  5. ST-LLM 处理:
  # 组合所有嵌入
  data_st = torch.cat([enc.permute(0, 3, 2, 1)] + [tem_emb] + node_emb, dim=1)
  data_st = self.in_layer(data_st)
  data_st = F.leaky_relu(data_st)
  data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)

  # 使用 GPT-2 处理
  adj = self.adj_mx_dict[select_dataset].to(self.device)
  outputs = self.gpt(data_st, adj)
  6. 输出生成:
  outputs = outputs.permute(0, 2, 1).unsqueeze(-1)
  outputs = self.regression_layer(outputs)
  skip = outputs.permute(0, 3, 2, 1)
  7. 反归一化:
  skip = skip * stdev
  skip = skip + means

  关键创新点

  1. ST-LLM-Plus 集成:
    - 使用 LoRA 的参数高效微调
    - 选择性层冻结 (仅顶层 U 层可训练)
    - 将邻接矩阵集成作为注意力掩码
  2. 多模态嵌入:
    - 组合流量 patch、时间上下文、拉普拉斯位置、节点和时间嵌入
    - 全面的特征表示
  3. 混合图神经网络:
    - 结合 GCN 和 GAT 进行空间特征提取
    - 残差连接以改善训练效果

  训练过程

  1. 数据预处理和归一化
  2. 使用输入/输出窗口创建序列 (通常为 288 个时间步)
  3. 使用掩码 MAE 计算损失
  4. 使用 Adam 优化器进行反向传播

  该模型展示了传统时空建模技术与现代基于 LLM 的方法的复杂集成，使其非常适合处理复杂的交通预测任务。


