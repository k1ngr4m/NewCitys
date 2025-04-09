import torch
import torch.nn as nn
from NewCity import NewCity
from NewCity_Ablation_Weather import NewCity_Ablation_Weather
from NewCity_Ablation_Time import NewCity_Ablation_Time
from lib.metrics import MSE_torch

# 初始化参数、数据集等
args = ...
dataset_use = ...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dim_in = ...

# 初始化模型
model_full = NewCity(args, dataset_use, device, dim_in).to(device)
model_ablation_weather = NewCity_Ablation_Weather(args, dataset_use, device, dim_in).to(device)
model_ablation_time = NewCity_Ablation_Time(args, dataset_use, device, dim_in).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer_full = torch.optim.Adam(model_full.parameters(), lr=args.lr)
optimizer_ablation_weather = torch.optim.Adam(model_ablation_weather.parameters(), lr=args.lr)
optimizer_ablation_time = torch.optim.Adam(model_ablation_time.parameters(), lr=args.lr)

# 训练和评估函数
def train_and_evaluate(model, optimizer, train_loader, test_loader):
    for epoch in range(args.epochs):
        model.train()
        for input, lbls, select_dataset in train_loader:
            input = input.to(device)
            lbls = lbls.to(device)
            optimizer.zero_grad()
            output = model(input, lbls, select_dataset)
            loss = criterion(output, lbls)
            loss.backward()
            optimizer.step()

    model.eval()
    total_mse = 0
    with torch.no_grad():
        for input, lbls, select_dataset in test_loader:
            input = input.to(device)
            lbls = lbls.to(device)
            output = model(input, lbls, select_dataset)
            mse, _ = MSE_torch(output, lbls)
            total_mse += mse.item()
    return total_mse / len(test_loader)

# 训练和评估全模型
mse_full = train_and_evaluate(model_full, optimizer_full, train_loader, test_loader)
# 训练和评估消融天气数据的模型
mse_ablation_weather = train_and_evaluate(model_ablation_weather, optimizer_ablation_weather, train_loader, test_loader)
# 训练和评估消融时空上下文编码的模型
mse_ablation_time = train_and_evaluate(model_ablation_time, optimizer_ablation_time, train_loader, test_loader)

# 打印结果
print(f"Full model MSE: {mse_full}")
print(f"Ablation weather model MSE: {mse_ablation_weather}")
print(f"Ablation time model MSE: {mse_ablation_time}")