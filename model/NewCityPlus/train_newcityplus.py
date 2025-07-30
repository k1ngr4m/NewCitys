import torch
import numpy as np
import argparse
import os
import sys
import time
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import configparser

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.metrics import All_Metrics
from lib.data_process import load_st_dataset
from model.NewCityPlus.args import parse_args
from model.Model import Traffic_model

def load_config(config_file):
    """加载配置文件"""
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def prepare_dataloader(data, batch_size, shuffle=True):
    """准备数据加载器"""
    dataset = TensorDataset(torch.FloatTensor(data[0]), torch.FloatTensor(data[1]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def train_model(args, model, train_loader, val_loader, device):
    """训练模型"""
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.L1Loss()  # MAE损失
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
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
        
        print(f'Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f'./model_weights/NewCityPlus/best_model_nyc_taxi.pth')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
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
    mae, mape, rmse = All_Metrics(preds, truths, args.mae_thresh, args.mape_thresh)
    print(f'Test MAE: {mae:.4f}, MAPE: {mape:.4f}, RMSE: {rmse:.4f}')
    
    return mae, mape, rmse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./conf/NewCityPlus/NewCityPlus_NYC_TAXI.conf', 
                        help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', 
                        help='设备')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--mae_thresh', type=float, default=0.1, help='MAE阈值')
    parser.add_argument('--mape_thresh', type=float, default=0.1, help='MAPE阈值')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # 加载数据
    print('Loading data...')
    # 这里需要根据实际数据格式进行调整
    # 假设数据已经预处理并保存为numpy数组
    
    # 创建模型
    print('Creating model...')
    # 这里需要根据实际参数设置创建模型
    model = None  # 需要根据实际模型结构创建
    
    # 移动模型到设备
    if model is not None:
        model.to(device)
        print(f'Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters')
    
    # 训练模型
    if model is not None:
        print('Starting training...')
        # train_model(args, model, train_loader, val_loader, device)
    
    # 评估模型
    # evaluate_model(model, test_loader, device)
    
    print('Training completed!')

if __name__ == "__main__":
    main()