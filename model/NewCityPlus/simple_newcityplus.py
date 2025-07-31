import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNewCityPlus(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=64, num_nodes=263):
        super(SimpleNewCityPlus, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # 简化的编码器
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # 简化的解码器
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # 批归一化
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, input, lbls, select_dataset):
        # input shape: [batch_size, time_steps, nodes, features]
        batch_size, time_steps, nodes, features = input.shape
        
        # 简化处理：只使用第一个节点的数据
        x = input[:, :, 0, :]  # [batch_size, time_steps, features]
        
        # 编码器
        encoder_out, (hidden, cell) = self.encoder(x)
        
        # 批归一化
        encoder_out = encoder_out.transpose(1, 2)  # [batch_size, hidden_dim, time_steps]
        encoder_out = self.batch_norm(encoder_out)
        encoder_out = encoder_out.transpose(1, 2)  # [batch_size, time_steps, hidden_dim]
        
        # 解码器
        decoder_out, _ = self.decoder(encoder_out, (hidden, cell))
        
        # 输出层
        output = self.output_layer(decoder_out)
        
        # 调整输出形状以匹配期望的格式
        output = output.unsqueeze(2)  # [batch_size, time_steps, 1, output_dim]
        output = output.expand(-1, -1, nodes, -1)  # [batch_size, time_steps, nodes, output_dim]
        
        return output

def create_simple_model():
    """创建简化版NewCityPlus模型"""
    model = SimpleNewCityPlus()
    return model

if __name__ == "__main__":
    # 测试模型
    model = create_simple_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # 创建测试输入
    batch_size, time_steps, nodes, features = 2, 12, 263, 3
    test_input = torch.randn(batch_size, time_steps, nodes, features)
    test_labels = torch.randn(batch_size, time_steps, nodes, 3)
    
    # 前向传播
    output = model(test_input, test_labels, 'NYC_TAXI')
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")