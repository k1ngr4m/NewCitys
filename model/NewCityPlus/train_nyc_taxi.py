import torch
import numpy as np
import argparse
import os
import sys
import time
from torch.utils.data import DataLoader, TensorDataset
import configparser

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.metrics import All_Metrics
from model.NewCityPlus.args import parse_args
from model.Model import Traffic_model

class Args:
    """参数类"""
    def __init__(self):
        self.dataset_use = ['NYC_TAXI']
        self.num_nodes_dict = {'NYC_TAXI': 263}
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = 'NewCityPlus'
        self.mode = 'train'
        self.input_base_dim = 3
        self.output_dim = 1
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
            './data/NYC_TAXI/NYC_TAXI.npz',
            './data/NYC_TAXI/train.npz',
            './data/NYC_TAXI/processed_data.npz'
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                data_path = file_path
                break
        else:
            raise FileNotFoundError("NYC_TAXI data file not found")
    
    # 加载数据
    if data_path.endswith('.npz'):
        data = np.load(data_path)
        if 'data' in data:
            raw_data = data['data']
        elif 'x' in data and 'y' in data:
            # 如果是已经分割好的训练数据
            x_data, y_data = data['x'], data['y']
            return x_data, y_data
        else:
            raise ValueError("Unknown data format in npz file")
    else:
        raise ValueError("Unsupported data format")
    
    # 如果是原始数据，需要进行处理
    print(f"Raw data shape: {raw_data.shape}")
    
    # 简单的数据处理示例
    # 假设数据形状为 [time_steps, nodes, features]
    time_steps, nodes, features = raw_data.shape
    
    # 创建序列数据
    input_window = 288
    output_window = 288
    stride = 12  # 时间步长
    
    xs, ys = [], []
    for i in range(0, time_steps - input_window - output_window, stride):
        x = raw_data[i:i+input_window]  # 输入序列
        y = raw_data[i+input_window:i+input_window+output_window]  # 输出序列
        xs.append(x)
        ys.append(y)
    
    xs = np.array(xs)
    ys = np.array(ys)
    
    print(f"Processed data - X shape: {xs.shape}, Y shape: {ys.shape}")
    return xs, ys

def prepare_dataloaders(x_data, y_data, batch_size=32, train_ratio=0.7, val_ratio=0.1):
    """准备数据加载器"""
    total_samples = len(x_data)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
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

def train_model(model, train_loader, val_loader, device, epochs=100, learning_rate=0.001, patience=10):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = torch.nn.L1Loss()  # MAE损失
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data, target, 'NYC_TAXI')
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data, target, 'NYC_TAXI')
                loss = criterion(output, target)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 创建模型权重目录
            os.makedirs('./model_weights/NewCityPlus', exist_ok=True)
            # 保存最佳模型
            torch.save(model.state_dict(), './model_weights/NewCityPlus/best_model_nyc_taxi.pth')
            print(f'Best model saved with validation loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    preds = []
    truths = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data, target, 'NYC_TAXI')
            preds.append(output.cpu().numpy())
            truths.append(target.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    truths = np.concatenate(truths, axis=0)
    
    # 计算评估指标
    # 简单的MAE, MAPE, RMSE计算
    mae = np.mean(np.abs(preds - truths))
    mape = np.mean(np.abs((preds - truths) / (truths + 1e-8))) * 100
    rmse = np.sqrt(np.mean((preds - truths) ** 2))
    
    print(f'Test Results - MAE: {mae:.4f}, MAPE: {mape:.2f}%, RMSE: {rmse:.4f}')
    
    return mae, mape, rmse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/NYC_TAXI/NYC_TAXI.npz', 
                        help='NYC_TAXI数据文件路径')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', 
                        help='设备')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    
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
    train_loader, val_loader, test_loader = prepare_dataloaders(
        x_data, y_data, batch_size=args.batch_size
    )
    
    # 创建模型参数
    model_args = Args()
    
    # 解析模型参数
    parser = argparse.ArgumentParser()
    args_predictor = parse_args(parser, model_args)
    
    # 创建模型
    print('Creating NewCityPlus model...')
    model = Traffic_model(model_args, args_predictor)
    
    # 移动模型到设备
    model.to(device)
    print(f'Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters')
    
    # 训练模型
    print('Starting training...')
    train_model(
        model, train_loader, val_loader, device,
        epochs=args.epochs, learning_rate=args.learning_rate, patience=args.patience
    )
    
    # 加载最佳模型进行评估
    try:
        model.load_state_dict(torch.load('./model_weights/NewCityPlus/best_model_nyc_taxi.pth'))
        print('Evaluating best model...')
        evaluate_model(model, test_loader, device)
    except Exception as e:
        print(f"Error during evaluation: {e}")
    
    print('Training completed!')

if __name__ == "__main__":
    main()