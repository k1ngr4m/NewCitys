概述

  通过对比 NewCity 和 NewCityPlus 的代码实现，我们可以发现 NewCityPlus 删除了 MultiHeadCrossAttention 类。这一变化反映了模型架构的重大演进。

  主要原因

  1. 架构演进：引入 GPT-2 作为核心组件

  NewCity:
  - 使用自定义的时空注意力机制
  - 包含多个专门的注意力模块，如 MultiHeadCrossAttention

  NewCityPlus:
  - 完全重构架构，采用 GPT-2 作为核心组件
  - 通过 PFA（Probabilistic Fusion Adapter）类实现

  2. 信息融合方式的改变

  NewCity 中的跨注意力:
  class MultiHeadCrossAttention(nn.Module):
      def __init__(self, d_traffic, d_weather, d_h, num_heads):
          # 交通特征作为 Query，天气特征作为 Key/Value

  NewCityPlus 中的信息融合:
  # 合并所有嵌入（交通、时间、节点）
  data_st = torch.cat([enc.permute(0, 3, 2, 1)] + [tem_emb] + node_emb, dim=1)
  # 然后通过 GPT-2 处理
  outputs = self.gpt(data_st, adj)

  3. 模型设计理念不同

  | NewCity                   | NewCityPlus                     |
  |---------------------------|---------------------------------|
  | 使用多个专门的注意力模块分别处理不同类型的信息交互 | 采用统一的大语言模型（GPT-2）作为通用的信息处理和融合平台 |
  | 自定义的时空注意力机制               | 基于预训练 Transformer 的通用架构         |

  4. 具体实现差异

  NewCityPlus 通过以下方式实现多模态信息融合：
  1. 将交通数据、时间嵌入、节点嵌入统一拼接
  2. 通过输入层处理后输入到 GPT-2
  3. 利用 GPT-2 的自注意力机制自动学习模态间关系

  优势

  简化架构

  - 不再需要专门的跨注意力模块
  - 统一的处理流程，减少模块间耦合

  更强的表示能力

  - 利用预训练的 GPT-2 模型的强大表示能力
  - 受益于大规模预训练的知识

  更好的泛化能力

  - 统一的处理框架可以适应更多模态的信息
  - 更容易扩展到其他类型的数据

  参数效率

  - 通过 LoRA 等技术进行参数高效微调
  - 相比维护多个专门模块更加参数高效

  总结

  NewCityPlus 删除 MultiHeadCrossAttention 是因为采用了全新的基于大语言模型的架构设计。它用 GPT-2
  的自注意力机制来统一处理和融合多种模态的信息，而不是使用多个专门的注意力模块。这种设计更加简洁且具有更强的表示能力，代表了时空预测模型向大模型范式转变的趋势。

