import h5py
import numpy as np
from datetime import datetime

# 加载流量数据
flow_file = 'BJ16_M32x32_T30_InOut.h5'
with h5py.File(flow_file, 'r') as f:
    flow_dates = f['date'][:]  # 日期数据
    flow_data = f['data'][:]  # 流量数据，shape: (num_timeslots, 2, 32, 32)

# 打印前几个日期，检查格式
print("Flow dates sample:", flow_dates[:5])

# 过滤并解析日期
valid_flow_dates = []
valid_flow_data = []
for i, date in enumerate(flow_dates):
    date_str = date.decode('utf-8')
    try:
        # 解析日期，格式为 '%Y%m%d%H'
        date_obj = datetime.strptime(date_str, '%Y%m%d%H')
        # 检查小时部分是否在 00-23 之间
        if 0 <= date_obj.hour < 24:
            valid_flow_dates.append(date_obj)
            valid_flow_data.append(flow_data[i])
    except ValueError:
        print(f"跳过无效日期: {date_str}")

# 加载天气数据
meteo_file = 'BJ_Meteorology.h5'
with h5py.File(meteo_file, 'r') as f:
    meteo_dates = f['date'][:]  # 日期数据
    temperature = f['Temperature'][:]  # 温度数据
    wind_speed = f['WindSpeed'][:]  # 风速数据
    weather = f['Weather'][:]  # 天气数据，shape: (num_timeslots, 17)

# 将天气日期转换为 datetime 对象
meteo_dates = [datetime.strptime(date.decode('utf-8'), '%Y%m%d%H%M') for date in meteo_dates]
# 过滤并解析日期
valid_flow_dates = []
valid_flow_data = []
for i, date in enumerate(flow_dates):
    date_str = date.decode('utf-8')
    try:
        # 解析日期，格式为 '%Y%m%d%H'
        date_obj = datetime.strptime(date_str, '%Y%m%d%H')
        # 检查小时部分是否在 00-23 之间
        if 0 <= date_obj.hour < 24:
            valid_flow_dates.append(date_obj)
            valid_flow_data.append(flow_data[i])
    except ValueError:
        print(f"跳过无效日期: {date_str}")
# 创建一个字典来存储天气数据，以便快速查找
meteo_dict = {date: (temp, wind, wthr) for date, temp, wind, wthr in zip(meteo_dates, temperature, wind_speed, weather)}

# 初始化融合后的数据
num_timeslots = len(valid_flow_dates)
num_nodes = 32
num_features = 5  # 2 (flow) + 3 (meteo)
merged_data = np.zeros((num_timeslots, num_nodes, num_nodes, num_features))

# 遍历流量数据，合并天气数据
for i, date in enumerate(valid_flow_dates):
    if date in meteo_dict:
        temp, wind, wthr = meteo_dict[date]
        # 获取流量数据
        inflow = valid_flow_data[i][0, :, :]
        outflow = valid_flow_data[i][1, :, :]
        # 合并数据
        merged_data[i, :, :, 0] = inflow
        merged_data[i, :, :, 1] = outflow
        merged_data[i, :, :, 2] = temp
        merged_data[i, :, :, 3] = wind
        # 将天气类型转换为单一值（取最大值索引）
        merged_data[i, :, :, 4] = np.argmax(wthr)

# 保存为 .npz 文件
np.savez('TaxiBJ.npz', data=merged_data)

print("数据融合完成并保存为 TaxiBJ.npz")

# 验证输出形状
print("转换后的数据形状:", merged_data.shape)