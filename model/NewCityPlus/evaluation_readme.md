# 模型评估和可视化工具使用说明

## 目录结构
- `evaluate_simple_newcityplus.py`: 独立评估脚本
- `train_simple_newcityplus.py`: 训练脚本（包含评估功能）
- `visualize_results.py`: 结果可视化脚本
- `evaluation_summary.md`: 评估结果总结文档

## 使用方法

### 1. 运行独立评估
```bash
python model/NewCityPlus/evaluate_simple_newcityplus.py
```

### 2. 运行训练脚本的评估模式
```bash
python model/NewCityPlus/train_simple_newcityplus.py --evaluate_only --model_path ./model_weights/SimpleNewCityPlus/model_epoch_2.pth
```

### 3. 生成可视化图表
```bash
python model/NewCityPlus/visualize_results.py
```

## 输出文件

### 数据文件
- `test_results/test_results.npy`: 测试结果数据
- `evaluation_results/evaluation_results.npy`: 评估结果数据

### 可视化图表
- `visualization/prediction_scatter.png`: 预测值vs真实值散点图
- `visualization/time_series_predictions.png`: 时间序列预测示例图
- `visualization/error_distribution.png`: 误差分布直方图
- `visualization/metrics_bar_chart.png`: 模型评估指标柱状图

## 评估指标说明
- **MAE (Mean Absolute Error)**: 平均绝对误差，衡量预测值与真实值之间的平均绝对差异
- **RMSE (Root Mean Square Error)**: 均方根误差，对较大误差给予更高权重
- **MAPE (Mean Absolute Percentage Error)**: 平均绝对百分比误差，以百分比形式表示预测精度