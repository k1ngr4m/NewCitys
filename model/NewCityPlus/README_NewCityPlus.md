# NewCityPlus Model

## Overview

NewCityPlus is an advanced spatio-temporal traffic prediction model that extends the NewCity architecture with Large Language Model (LLM) integration. It's designed to forecast traffic patterns by combining graph neural networks with GPT-2 through parameter-efficient fine-tuning techniques.

## Architecture

### Core Components

1. **Patch Embedding**
   - Processes temporal data through patching mechanisms
   - Handles both flow data and time features

2. **Spatio-Temporal Attention**
   - Specialized attention mechanism for spatial and temporal dependencies
   - TemporalSelfAttention with time context (TC) and temporal (T) attention heads

3. **GPT-2 Integration**
   - Parameter-Efficient Fine-tuning (PFA) approach with LoRA
   - Selective layer freezing for efficient training

4. **Graph Neural Networks**
   - Combines GCN and GAT for spatial feature extraction
   - Adjacency matrix integration as attention masks

5. **Positional Encoding**
   - Laplacian positional encoding for spatial structure information

### Key Features

- **STEncoderBlock**: Stacked encoder blocks processing spatio-temporal features
- **Multi-Modal Embeddings**: Flow patches, time context, Laplacian positions, node, and temporal embeddings
- **Custom GPT-2 Forward Pass**: Incorporates graph structure into LLM processing

## Configuration

### Key Parameters

- `embed_dim`: Embedding dimension (128 for base, 256 for NYC_TAXI)
- `input_window/output_window`: Sequence lengths (typically 288 timesteps)
- `lape_dim`: Laplacian positional encoding dimension
- `llm_layer`: Number of GPT-2 layers to use (6)
- `U`: Number of unfrozen layers in GPT-2 (1 or 2 depending on dataset)

### Configuration Files

1. `conf/NewCityPlus/NewCityPlus.conf`: Base configuration
2. `conf/NewCityPlus/NewCityPlus_NYC_TAXI.conf`: NYC_TAXI-specific optimized configuration

## Usage

### Model Integration

The model is integrated through `model/Model.py`:

```python
elif self.model == 'NewCityPlus':
    from model.NewCityPlus.NewCityPlus import NewCityPlus
    self.predictor = NewCityPlus(args_predictor, args.dataset_use, args.device, dim_in)
```

### Data Format

Input data structure:
- Shape: `[batch_size, time_steps, num_nodes, features]`
- Features include traffic flow and temporal information
- Adjacency matrices for spatial relationships

### Training Process

1. Data preprocessing with normalization
2. Sequence creation with input/output windowing
3. Loss computation using masked MAE
4. Backpropagation with Adam optimizer

## Key Innovations

### ST-LLM-Plus Integration
- Parameter-efficient fine-tuning with LoRA
- Selective layer freezing (only top U layers trainable)
- Adjacency matrix integration as attention masks

### Multi-Modal Embeddings
Combines multiple embedding types for comprehensive feature representation.