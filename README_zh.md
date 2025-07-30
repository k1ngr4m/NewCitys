# NewCity：用于交通预测的时空基础模型

## 简介

NewCity是一个先进的时空基础模型，用于交通预测，它基于OpenCity框架构建。该模型结合了复杂的注意力机制和图神经网络，能够有效建模交通数据中的复杂时空依赖关系。

NewCity的关键特性包括：
- **多头时序自注意力机制**：使用专门的注意力头捕捉时间模式
- **图注意力网络(GAT)**：使用图注意力机制建模空间依赖关系
- **拉普拉斯位置编码**：融入交通网络的结构信息
- **基于补丁的嵌入**：以补丁方式处理交通数据以提高学习效率
- **天气数据集成**：可选择性地整合天气数据以增强预测效果

## 模型架构

NewCity模型由几个关键组件组成：

1. **补丁嵌入**：将交通数据分割成补丁以提高处理效率
2. **时序上下文编码**：编码时间信息，包括一天中的时间和星期几
3. **空间编码**：使用拉普拉斯位置编码捕捉空间关系
4. **时空编码器**：堆叠的Transformer块，包含GAT和GCN层
5. **预测头**：生成最终的交通预测结果

## 配置

模型配置在`conf/NewCity/NewCity.conf`中定义：

```ini
[data]
input_window = 288
output_window = 288

[model]
embed_dim = 128
skip_dim = 128
lape_dim = 8
geo_num_heads = 0
sem_num_heads = 0
tc_num_heads = 16
t_num_heads = 16
mlp_ratio = 2
qkv_bias = True
drop = 0.1
attn_drop = 0.3
drop_path = 0.0
s_attn_size = 3
t_attn_size = 1
enc_depth = 3
type_ln = pre
type_short_path = hop
far_mask_delta = 5
weather_dim = 0

[train]
seed = 12
seed_mode = False
xavier = False
loss_func = mask_mae
real_value = True
```

## 使用方法

### 环境设置

```bash
conda create -n newcity python=3.9.13
conda activate newcity

# 安装PyTorch（根据您的系统调整）
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# 安装所需库
pip install -r requirements.txt
```

### 训练

训练NewCity模型：

```bash
# 基本训练
python Run.py -mode train -model NewCity

# 使用自定义配置训练
python Run.py -mode train -model NewCity -batch_size 8 --embed_dim 256 --skip_dim 256 --enc_depth 3
```

### 评估

评估NewCity模型：

```bash
# 使用预训练权重进行评估
python Run.py -mode test -model NewCity -load_pretrain_path newcity_weights.pth
```

## 核心组件

### 1. 时序自注意力机制
`TemporalSelfAttention`模块使用时序和时序上下文注意力机制来捕捉复杂的时间依赖关系。

### 2. 图神经网络
NewCity结合了图卷积网络(GCN)和图注意力网络(GAT)进行空间建模。

### 3. 补丁嵌入
使用基于补丁的嵌入处理交通数据，以提高学习效率并降低计算复杂度。

### 4. 拉普拉斯位置编码
使用拉普拉斯特征向量对交通网络的结构信息进行编码。

## 数据格式

模型期望的交通数据格式如下：
- 形状：`[batch_size, time_steps, num_nodes, features]`
- 特征通常包括交通流量、时间信息，以及可选的天气数据

## 自定义

要为您的特定用例定制模型：

1. 在`conf/NewCity/NewCity.conf`中调整配置参数
2. 在`model/NewCity/NewCity.py`中修改模型架构
3. 在`lib/data_process.py`中更新数据处理逻辑

## 结果

NewCity在各种交通预测基准测试中表现出色，展示了：
- 卓越的零样本泛化能力
- 对多样化交通模式的鲁棒处理
- 空间和时间信息的有效整合

## 引用

如果您在研究中使用NewCity，请引用以下论文：

```bibtex
@misc{li2024opencity,
      title={OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction}, 
      author={Zhonghang Li and Long Xia and Lei Shi and Yong Xu and Dawei Yin and Chao Huang},
      year={2024},
      eprint={2408.10269},
      archivePrefix={arXiv}
}
```

## 致谢

NewCity基于OpenCity框架构建，并融合了各种时空建模方法的理念。