# NewCityPlus Model

NewCityPlus is an enhanced version of the NewCity model that incorporates key features from the ST-LLM-Plus model. It combines the spatio-temporal modeling capabilities of NewCity with the powerful language model backbone of ST-LLM-Plus.

## Key Features

1. **Hybrid Architecture**: Combines NewCity's spatio-temporal attention mechanisms with ST-LLM-Plus's GPT-2 backbone
2. **LoRA Fine-tuning**: Utilizes Low-Rank Adaptation for efficient parameter tuning
3. **Multi-modal Embeddings**: Integrates traffic flow, temporal, and spatial embeddings
4. **Graph-aware Attention**: Incorporates adjacency matrix information into the attention mechanism

## Architecture Overview

NewCityPlus consists of the following key components:

1. **Patch Embedding**: Processes traffic data in patches for efficient learning
2. **Temporal Context Encoding**: Encodes time-of-day and day-of-week information
3. **Spatial Encoding**: Uses Laplacian positional encoding for spatial relationships
4. **Spatio-Temporal Encoder**: Combines NewCity's attention mechanisms with ST-LLM-Plus's GPT-2 layers
5. **GPT-2 Backbone**: Utilizes a pretrained GPT-2 model with LoRA adaptation
6. **Prediction Head**: Generates final traffic predictions

## ST-LLM-Plus Integration

NewCityPlus incorporates several key features from ST-LLM-Plus:

1. **PFA (Parametric Fusion Attention)**: Custom attention mechanism that integrates graph structure
2. **Temporal Embedding**: Dedicated temporal feature encoding
3. **Node Embedding**: Learnable node-specific embeddings
4. **LoRA Configuration**: Efficient parameter adaptation using Low-Rank Adaptation
5. **Layer-freezing Strategy**: Selective parameter updating for efficiency

## NYC_TAXI Dataset Special Processing

For the NYC_TAXI dataset, NewCityPlus includes special processing:

1. **Data Preprocessing**: Custom script to handle NYC_TAXI data format
2. **Adjacency Matrix Handling**: Proper integration of NYC_TAXI's road network structure
3. **Optimized Configuration**: Specialized hyperparameters for NYC_TAXI characteristics
4. **Sequence Processing**: Efficient handling of long time series data

## Configuration

The model can be configured using the following files:
- `conf/NewCityPlus/NewCityPlus.conf`: General configuration
- `conf/NewCityPlus/NewCityPlus_NYC_TAXI.conf`: NYC_TAXI-specific configuration

Key parameters include:
- `embed_dim`: Embedding dimension
- `llm_layer`: Number of GPT-2 layers to use
- `U`: Number of unfrozen layers
- `enc_depth`: Depth of the encoder

## Usage

### Training

To train the NewCityPlus model on NYC_TAXI data:

```bash
python model/NewCityPlus/train_nyc_taxi.py --data_path ./data/NYC_TAXI/NYC_TAXI.npz --epochs 50 --batch_size 16
```

### Data Preprocessing

To preprocess the NYC_TAXI dataset:

```bash
cd data/NYC_TAXI
python generate_nyc_taxi_data.py --task all
```

## Model Files

- `NewCityPlus.py`: Main model implementation
- `args.py`: Argument parsing and configuration
- `train_nyc_taxi.py`: Training script for NYC_TAXI dataset
- `train_newcityplus.py`: General training script

## Requirements

NewCityPlus requires the following additional packages:
- `transformers`
- `peft`

Install with:
```bash
pip install transformers peft
```