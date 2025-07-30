import numpy as np
import os
import argparse
from sklearn.preprocessing import StandardScaler

def generate_nyc_taxi_dataset():
    """
    生成适用于NewCityPlus模型的NYC_TAXI数据集
    """
    # 加载原始数据
    data_path = './NYC_TAXI.npz'
    adj_path = './NYC_TAXI_rn_adj.npy'
    
    if not os.path.exists(data_path):
        print(f"数据文件 {data_path} 不存在")
        return
    
    # 加载数据
    raw_data = np.load(data_path)
    data = raw_data['data']
    print(f"原始数据形状: {data.shape}")
    
    # 加载邻接矩阵
    adj_matrix = np.load(adj_path)
    print(f"邻接矩阵形状: {adj_matrix.shape}")
    
    # 数据预处理
    # NYC_TAXI数据集包含两个特征：[流量, 时间信息]
    # 我们需要将其转换为模型期望的格式
    flow_data = data[:, :, 0]  # 流量数据
    time_data = data[:, :, 1]  # 时间信息
    
    # 数据标准化
    scaler = StandardScaler()
    flow_data_scaled = scaler.fit_transform(flow_data)
    
    # 重新组合数据
    # 为了适应模型，我们需要添加时间特征
    processed_data = np.zeros((flow_data.shape[0], flow_data.shape[1], 3))
    processed_data[:, :, 0] = flow_data_scaled  # 标准化后的流量
    processed_data[:, :, 1] = time_data  # 时间信息
    processed_data[:, :, 2] = np.zeros_like(time_data)  # 额外的特征维度（可填充其他信息）
    
    print(f"处理后数据形状: {processed_data.shape}")
    
    # 保存处理后的数据
    np.savez('./NYC_TAXI_processed.npz', data=processed_data)
    print("数据预处理完成，已保存为 NYC_TAXI_processed.npz")
    
    # 保存标准化参数以供后续反标准化使用
    np.save('./scaler_params.npy', {'mean': scaler.mean_, 'scale': scaler.scale_})
    print("标准化参数已保存为 scaler_params.npy")

def create_adjacency_matrix():
    """
    创建或验证邻接矩阵
    """
    adj_path = './NYC_TAXI_rn_adj.npy'
    if os.path.exists(adj_path):
        adj_matrix = np.load(adj_path)
        print(f"邻接矩阵已存在，形状: {adj_matrix.shape}")
        
        # 确保对角线为1（自连接）
        np.fill_diagonal(adj_matrix, 1)
        
        # 保存更新后的邻接矩阵
        np.save('./NYC_TAXI_rn_adj_updated.npy', adj_matrix)
        print("邻接矩阵已更新并保存为 NYC_TAXI_rn_adj_updated.npy")
    else:
        print("邻接矩阵文件不存在")

def prepare_dataset_for_training():
    """
    准备训练数据集
    """
    # 加载处理后的数据
    processed_data_path = './NYC_TAXI_processed.npz'
    if not os.path.exists(processed_data_path):
        print("请先运行数据预处理")
        return
    
    data = np.load(processed_data_path)['data']
    print(f"加载处理后的数据，形状: {data.shape}")
    
    # 划分数据集
    total_time_steps = data.shape[0]
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2
    
    train_end = int(total_time_steps * train_ratio)
    val_end = int(total_time_steps * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"训练集大小: {train_data.shape}")
    print(f"验证集大小: {val_data.shape}")
    print(f"测试集大小: {test_data.shape}")
    
    # 创建序列数据
    input_window = 288  # 输入序列长度
    output_window = 288  # 输出序列长度
    
    def create_sequences(dataset):
        xs, ys = [], []
        for i in range(len(dataset) - input_window - output_window):
            x = dataset[i:(i + input_window)]
            y = dataset[(i + input_window):(i + input_window + output_window)]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    train_x, train_y = create_sequences(train_data)
    val_x, val_y = create_sequences(val_data)
    test_x, test_y = create_sequences(test_data)
    
    print(f"训练序列形状: x={train_x.shape}, y={train_y.shape}")
    print(f"验证序列形状: x={val_x.shape}, y={val_y.shape}")
    print(f"测试序列形状: x={test_x.shape}, y={test_y.shape}")
    
    # 保存数据集
    np.savez('./train.npz', x=train_x, y=train_y)
    np.savez('./val.npz', x=val_x, y=val_y)
    np.savez('./test.npz', x=test_x, y=test_y)
    
    print("数据集已保存为 train.npz, val.npz, test.npz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='all', 
                        choices=['preprocess', 'adjacency', 'prepare', 'all'],
                        help='要执行的任务')
    args = parser.parse_args()
    
    if args.task == 'preprocess' or args.task == 'all':
        print("执行数据预处理...")
        generate_nyc_taxi_dataset()
        
    if args.task == 'adjacency' or args.task == 'all':
        print("处理邻接矩阵...")
        create_adjacency_matrix()
        
    if args.task == 'prepare' or args.task == 'all':
        print("准备训练数据集...")
        prepare_dataset_for_training()
        
    print("NYC_TAXI数据集准备完成")