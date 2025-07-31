import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_visualize_results():
    """加载并可视化评估结果"""
    
    # 创建可视化目录
    os.makedirs('./visualization', exist_ok=True)
    
    # 加载测试结果
    try:
        test_results = np.load('./test_results/test_results.npy', allow_pickle=True).item()
        print("测试结果加载成功")
    except Exception as e:
        print(f"加载测试结果时出错: {e}")
        return
    
    # 提取结果
    mae = test_results['mae']
    rmse = test_results['rmse']
    mape = test_results['mape']
    predictions = test_results['predictions']
    targets = test_results['targets']
    
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE: {mape:.2f}%")
    
    # 可视化1: 预测值vs真实值散点图
    plt.figure(figsize=(10, 8))
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # 采样以避免点太多
    sample_indices = np.random.choice(len(pred_flat), min(10000, len(pred_flat)), replace=False)
    pred_sample = pred_flat[sample_indices]
    target_sample = target_flat[sample_indices]
    
    plt.scatter(target_sample, pred_sample, alpha=0.5, s=1)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测值 vs 真实值散点图')
    
    # 添加y=x线
    min_val = min(np.min(target_sample), np.min(pred_sample))
    max_val = max(np.max(target_sample), np.max(pred_sample))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./visualization/prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 可视化2: 时间序列预测示例
    plt.figure(figsize=(15, 10))
    
    # 选择几个样本进行可视化
    num_samples = min(4, len(predictions))
    sample_indices = np.random.choice(len(predictions), num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 2, i+1)
        
        # 选择第一个节点和第一个特征进行可视化
        target_ts = targets[idx, :, 0, 0]  # [time_steps]
        pred_ts = predictions[idx, :, 0, 0]  # [time_steps]
        
        time_steps = range(len(target_ts))
        plt.plot(time_steps, target_ts, 'b-', label='真实值', linewidth=2)
        plt.plot(time_steps, pred_ts, 'r--', label='预测值', linewidth=2)
        
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.title(f'样本 {idx+1} 时间序列预测')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./visualization/time_series_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 可视化3: 误差分布直方图
    plt.figure(figsize=(10, 6))
    errors = target_flat - pred_flat
    plt.hist(errors, bins=100, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    plt.xlabel('误差')
    plt.ylabel('频率')
    plt.title('预测误差分布直方图')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./visualization/error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 可视化4: 指标柱状图
    plt.figure(figsize=(8, 6))
    metrics = ['MAE', 'RMSE', 'MAPE']
    values = [mae, rmse, mape]
    
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylabel('值')
    plt.title('模型评估指标')
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, values):
        if bar.get_x() == 2:  # MAPE
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{value:.2f}%', 
                    ha='center', va='bottom', fontsize=12)
        else:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{value:.4f}', 
                    ha='center', va='bottom', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./visualization/metrics_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("可视化结果已保存到 ./visualization/ 目录")

def main():
    load_and_visualize_results()

if __name__ == "__main__":
    main()