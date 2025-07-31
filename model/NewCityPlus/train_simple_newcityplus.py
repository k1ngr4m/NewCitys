import torch
import numpy as np
import argparse
import os
import sys
from torch.utils.data import DataLoader, TensorDataset

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.NewCityPlus.simple_newcityplus import SimpleNewCityPlus

class Args:
    """参数类"""
    def __init__(self):
        self.dataset_use = ['NYC_TAXI']
        self.num_nodes_dict = {'NYC_TAXI': 263}
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = 'SimpleNewCityPlus'
        self.mode = 'train'
        self.input_base_dim = 2
        self.input_extra_dim = 1  # NYC_TAXI_processed有3个特征维度
        self.output_dim = 3  # 修改为3以匹配输入维度
        self.log_dir = './log'
        self.load_pretrain_path = ''
        self.dataset = 'NYC_TAXI'

def load_nyc_taxi_data(data_path):
    """加载NYC_TAXI数据"""
    print(f"Loading data from {data_path}")
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        # 尝试不同的数据文件名
        possible_files = [
            './data/NYC_TAXI/NYC_TAXI_processed.npz',
            './data/NYC_TAXI/NYC_TAXI.npz'
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                data_path = file_path
                break
        else:
            raise FileNotFoundError("NYC_TAXI data file not found")
    
    # 加载数据
    data = np.load(data_path)
    if 'data' in data:
        raw_data = data['data']
    else:
        raise ValueError("Unknown data format in npz file")
    
    # 如果是原始数据，需要进行处理
    print(f"Raw data shape: {raw_data.shape}")
    
    # 数据形状为 [nodes, time_steps, features]
    nodes, time_steps, features = raw_data.shape
    print(f"Data shape: nodes={nodes}, time_steps={time_steps}, features={features}")
    
    # 创建序列数据 - 使用较小的窗口以适应内存限制
    input_window = 12  # 减小窗口大小
    output_window = 12
    stride = 6  # 使用较小的步长
    
    xs, ys = [], []
    
    # 选择前几个节点进行训练以减少计算量
    num_nodes_to_use = min(10, nodes)  # 只使用前10个节点
    print(f"Using {num_nodes_to_use} nodes for training")
    
    for node_idx in range(num_nodes_to_use):
        node_data = raw_data[node_idx]  # [time_steps, features]
        # 确保时间步长足够长
        if time_steps < input_window + output_window:
            print(f"Skipping node {node_idx} due to insufficient time steps")
            continue
            
        for i in range(0, time_steps - input_window - output_window, stride):
            x = node_data[i:i+input_window]  # 输入序列 [input_window, features]
            y = node_data[i+input_window:i+input_window+output_window]  # 输出序列 [output_window, features]
            # 扩展维度以适应模型输入 [1, input_window, 1, features]
            x = np.expand_dims(np.expand_dims(x, axis=0), axis=2)  # [1, input_window, 1, features]
            y = np.expand_dims(np.expand_dims(y, axis=0), axis=2)  # [1, output_window, 1, features]
            xs.append(x)
            ys.append(y)
    
    if len(xs) == 0:
        raise ValueError("No training samples generated. Check data processing logic.")
    
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    
    print(f"Processed data - X shape: {xs.shape}, Y shape: {ys.shape}")
    return xs, ys

def prepare_dataloaders(x_data, y_data, batch_size=2, train_ratio=0.7, val_ratio=0.15):
    """准备数据加载器"""
    total_samples = len(x_data)
    if total_samples == 0:
        raise ValueError("No samples available for training")
        
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    # 确保至少有一个样本用于每个集合
    if train_end == 0:
        train_end = 1
    if val_end <= train_end:
        val_end = min(train_end + 1, total_samples)
    
    # 划分数据集
    x_train, y_train = x_data[:train_end], y_data[:train_end]
    x_val, y_val = x_data[train_end:val_end], y_data[train_end:val_end]
    x_test, y_test = x_data[val_end:], y_data[val_end:]
    
    print(f"Train set: {x_train.shape}, Val set: {x_val.shape}, Test set: {x_test.shape}")
    
    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, epochs=3, learning_rate=0.001):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = torch.nn.L1Loss()  # MAE损失
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            try:
                output = model(data, target, 'NYC_TAXI')
                
                # 确保输出和目标形状匹配
                if output.shape != target.shape:
                    # 调整输出形状以匹配目标
                    min_time_steps = min(output.shape[1], target.shape[1])
                    output = output[:, :min_time_steps, :, :]
                    target = target[:, :min_time_steps, :, :]
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                # 每5个批次打印一次进度
                if batch_idx % 5 == 0:
                    print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        if num_batches > 0:
            train_loss /= num_batches
        else:
            train_loss = 0
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}')
        
        # 保存模型
        os.makedirs('./model_weights/SimpleNewCityPlus', exist_ok=True)
        torch.save(model.state_dict(), f'./model_weights/SimpleNewCityPlus/model_epoch_{epoch+1}.pth')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/NYC_TAXI/NYC_TAXI_processed.npz', 
                        help='NYC_TAXI数据文件路径')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', 
                        help='设备')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # 加载数据
    print('Loading NYC_TAXI data...')
    try:
        x_data, y_data = load_nyc_taxi_data(args.data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 准备数据加载器
    print('Preparing data loaders...')
    try:
        train_loader, val_loader, test_loader = prepare_dataloaders(
            x_data, y_data, batch_size=args.batch_size
        )
    except Exception as e:
        print(f"Error preparing data loaders: {e}")
        return
    
    # 创建模型
    print('Creating SimpleNewCityPlus model...')
    model = SimpleNewCityPlus()
    
    # 移动模型到设备
    model.to(device)
    print(f'Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters')
    
    # 训练模型
    print('Starting training...')
    train_model(
        model, train_loader, val_loader, device,
        epochs=args.epochs, learning_rate=args.learning_rate
    )
    
    print('Training completed!')

if __name__ == "__main__":
    main()