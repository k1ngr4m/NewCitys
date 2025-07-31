import torch
import numpy as np
import argparse
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.NewCityPlus.simple_newcityplus import SimpleNewCityPlus

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
    print(f"Using {num_nodes_to_use} nodes for evaluation")
    
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
        raise ValueError("No evaluation samples generated. Check data processing logic.")
    
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    
    print(f"Processed data - X shape: {xs.shape}, Y shape: {ys.shape}")
    return xs, ys

def prepare_dataloaders(x_data, y_data, batch_size=2, train_ratio=0.7, val_ratio=0.15):
    """准备数据加载器"""
    total_samples = len(x_data)
    if total_samples == 0:
        raise ValueError("No samples available for evaluation")
        
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

def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            try:
                output = model(data, target, 'NYC_TAXI')
                
                # 确保输出和目标形状匹配
                if output.shape != target.shape:
                    # 调整输出形状以匹配目标
                    min_time_steps = min(output.shape[1], target.shape[1])
                    output = output[:, :min_time_steps, :, :]
                    target = target[:, :min_time_steps, :, :]
                
                # 收集预测结果和真实值
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    if len(predictions) == 0 or len(targets) == 0:
        raise ValueError("No predictions generated during evaluation")
    
    # 合并所有批次的结果
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # 计算评估指标
    # 展平数组以计算整体指标
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # MAE
    mae = mean_absolute_error(target_flat, pred_flat)
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(target_flat, pred_flat))
    
    # MAPE (避免除以零)
    mask = target_flat != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((target_flat[mask] - pred_flat[mask]) / target_flat[mask])) * 100
    else:
        mape = np.mean(np.abs(pred_flat - target_flat)) * 100  # 如果所有真实值都为0，使用MAE
    
    return mae, rmse, mape, predictions, targets

def calculate_node_metrics(predictions, targets):
    """计算每个节点的评估指标"""
    # predictions shape: [samples, time_steps, nodes, features]
    # targets shape: [samples, time_steps, nodes, features]
    
    node_metrics = []
    
    for node_idx in range(predictions.shape[2]):  # 遍历所有节点
        node_pred = predictions[:, :, node_idx, :].flatten()
        node_target = targets[:, :, node_idx, :].flatten()
        
        mae = mean_absolute_error(node_target, node_pred)
        rmse = np.sqrt(mean_squared_error(node_target, node_pred))
        
        mask = node_target != 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((node_target[mask] - node_pred[mask]) / node_target[mask])) * 100
        else:
            mape = np.mean(np.abs(node_pred - node_target)) * 100
            
        node_metrics.append({
            'node': node_idx,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        })
    
    return node_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/NYC_TAXI/NYC_TAXI_processed.npz', 
                        help='NYC_TAXI数据文件路径')
    parser.add_argument('--model_path', type=str, default='./model_weights/SimpleNewCityPlus/model_epoch_2.pth',
                        help='训练好的模型权重路径')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', 
                        help='设备')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    
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
    
    # 加载模型权重
    print(f'Loading model weights from {args.model_path}...')
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    
    # 移动模型到设备
    model.to(device)
    print(f'Model loaded successfully')
    
    # 评估模型
    print('Evaluating model...')
    try:
        mae, rmse, mape, predictions, targets = evaluate_model(model, test_loader, device)
        
        print("\n" + "="*50)
        print("模型评估结果")
        print("="*50)
        print(f"总体指标:")
        print(f"  MAE:  {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAPE: {mape:.2f}%")
        print("="*50)
        
        # 计算每个节点的指标
        print("计算每个节点的评估指标...")
        node_metrics = calculate_node_metrics(predictions, targets)
        
        # 打印前10个节点的指标
        print("\n前10个节点的评估指标:")
        print("-" * 40)
        for i, metrics in enumerate(node_metrics[:10]):
            print(f"节点 {metrics['node']:2d}: MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, MAPE={metrics['mape']:.2f}%")
        
        # 计算平均节点指标
        avg_mae = np.mean([m['mae'] for m in node_metrics])
        avg_rmse = np.mean([m['rmse'] for m in node_metrics])
        avg_mape = np.mean([m['mape'] for m in node_metrics])
        
        print("\n" + "-" * 40)
        print(f"平均节点指标:")
        print(f"  MAE:  {avg_mae:.6f}")
        print(f"  RMSE: {avg_rmse:.6f}")
        print(f"  MAPE: {avg_mape:.2f}%")
        print("="*50)
        
        # 保存评估结果
        os.makedirs('./evaluation_results', exist_ok=True)
        eval_results = {
            'overall_metrics': {
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            },
            'node_metrics': node_metrics,
            'predictions': predictions,
            'targets': targets
        }
        
        np.save('./evaluation_results/evaluation_results.npy', eval_results)
        print(f"\n评估结果已保存到 ./evaluation_results/evaluation_results.npy")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return
    
    print('Evaluation completed!')

if __name__ == "__main__":
    main()