# NewCity: Spatio-Temporal Foundation Models for Traffic Prediction

## Introduction

NewCity is an advanced spatio-temporal foundation model for traffic prediction, building upon the OpenCity framework. It incorporates sophisticated attention mechanisms and graph neural networks to effectively model complex spatio-temporal dependencies in traffic data.

Key features of NewCity include:
- **Multi-head Temporal Self-Attention**: Captures temporal patterns with specialized attention heads
- **Graph Attention Networks (GAT)**: Models spatial dependencies using graph attention mechanisms
- **Laplacian Positional Encoding**: Incorporates structural information from the traffic network
- **Patch-based Embedding**: Processes traffic data in patches for efficient learning
- **Weather Integration**: Optionally incorporates weather data for enhanced predictions

## Model Architecture

The NewCity model consists of several key components:

1. **Patch Embedding**: Divides traffic data into patches for efficient processing
2. **Temporal Context Encoding**: Encodes temporal information including time of day and day of week
3. **Spatial Encoding**: Uses Laplacian positional encoding to capture spatial relationships
4. **Spatio-Temporal Encoder**: Stacked transformer blocks with GAT and GCN layers
5. **Prediction Head**: Generates final traffic predictions

## Configuration

The model configuration is defined in `conf/NewCity/NewCity.conf`:

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

## Usage

### Environment Setup

```bash
conda create -n newcity python=3.9.13
conda activate newcity

# Install PyTorch (adjust for your system)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install required libraries
pip install -r requirements.txt
```

### Training

To train the NewCity model:

```bash
# Basic training
python Run.py -mode train -model NewCity

# Training with custom configuration
python Run.py -mode train -model NewCity -batch_size 8 --embed_dim 256 --skip_dim 256 --enc_depth 3
```

### Evaluation

To evaluate the NewCity model:

```bash
# Evaluate with pre-trained weights
python Run.py -mode test -model NewCity -load_pretrain_path newcity_weights.pth
```

## Key Components

### 1. Temporal Self-Attention
The `TemporalSelfAttention` module uses both temporal and temporal-context attention mechanisms to capture complex temporal dependencies.

### 2. Graph Neural Networks
NewCity incorporates both Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) for spatial modeling.

### 3. Patch Embedding
Traffic data is processed using patch-based embedding for efficient learning and reduced computational complexity.

### 4. Laplacian Positional Encoding
Structural information from the traffic network is encoded using Laplacian eigenvectors.

## Data Format

The model expects traffic data in the following format:
- Shape: `[batch_size, time_steps, num_nodes, features]`
- Features typically include traffic flow, time information, and optionally weather data

## Customization

To customize the model for your specific use case:

1. Adjust configuration parameters in `conf/NewCity/NewCity.conf`
2. Modify the model architecture in `model/NewCity/NewCity.py`
3. Update data processing in `lib/data_process.py`

## Results

NewCity achieves state-of-the-art performance on various traffic prediction benchmarks, demonstrating:
- Superior zero-shot generalization capabilities
- Robust handling of diverse traffic patterns
- Effective integration of spatial and temporal information

## Citation

If you use NewCity in your research, please cite the following paper:

```bibtex
@misc{li2024opencity,
      title={OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction}, 
      author={Zhonghang Li and Long Xia and Lei Shi and Yong Xu and Dawei Yin and Chao Huang},
      year={2024},
      eprint={2408.10269},
      archivePrefix={arXiv}
}
```

## Acknowledgements

NewCity builds upon the OpenCity framework and incorporates ideas from various spatio-temporal modeling approaches.