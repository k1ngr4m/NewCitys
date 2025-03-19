# import h5py
# import numpy as np
# from datetime import datetime
#
# # 加载流量数据
# flow_file = 'BJ16_M32x32_T30_InOut.h5'
# with h5py.File(flow_file, 'r') as f:
#     flow_dates = f['date'][:]  # 日期数据
#     flow_data = f['data'][:]  # 流量数据，shape: (num_timeslots, 2, 32, 32)
#
# # 打印前几个日期，检查格式
# print("Flow dates sample:", flow_dates[:5])
#
# # 过滤并解析日期
# valid_flow_dates = []
# valid_flow_data = []
# for i, date in enumerate(flow_dates):
#     date_str = date.decode('utf-8')
#     try:
#         # 解析日期，格式为 '%Y%m%d%H'
#         date_obj = datetime.strptime(date_str, '%Y%m%d%H')
#         # 检查小时部分是否在 00-23 之间
#         if 0 <= date_obj.hour < 24:
#             valid_flow_dates.append(date_obj)
#             valid_flow_data.append(flow_data[i])
#     except ValueError:
#         print(f"跳过无效日期: {date_str}")
#
# # 加载天气数据
# meteo_file = 'BJ_Meteorology.h5'
# with h5py.File(meteo_file, 'r') as f:
#     meteo_dates = f['date'][:]  # 日期数据
#     temperature = f['Temperature'][:]  # 温度数据
#     wind_speed = f['WindSpeed'][:]  # 风速数据
#     weather = f['Weather'][:]  # 天气数据，shape: (num_timeslots, 17)
#
# # 将天气日期转换为 datetime 对象
# meteo_dates = [datetime.strptime(date.decode('utf-8'), '%Y%m%d%H%M') for date in meteo_dates]
#
# # 创建一个字典来存储天气数据，以便快速查找
# meteo_dict = {date: (temp, wind, wthr) for date, temp, wind, wthr in zip(meteo_dates, temperature, wind_speed, weather)}
#
# # 初始化融合后的数据
# num_timeslots = len(valid_flow_dates)
# num_nodes = 32 * 32  # 将二维网格展平为一维节点
# num_features = 5  # 2 (flow) + 3 (meteo)
#
# # 修改后的三维数组格式 (时间步长, 节点数, 特征值)
# merged_data = np.zeros((num_timeslots, num_nodes, num_features))
#
# # 遍历流量数据，合并天气数据
# for i, date in enumerate(valid_flow_dates):
#     if date in meteo_dict:
#         temp, wind, wthr = meteo_dict[date]
#         # 获取并展平流量数据（32x32 -> 1024）
#         inflow = valid_flow_data[i][0].flatten()
#         outflow = valid_flow_data[i][1].flatten()
#
#         # 合并特征数据（注意天气数据需要广播到所有节点）
#         merged_data[i, :, 0] = inflow  # 流入量特征
#         merged_data[i, :, 1] = outflow  # 流出量特征
#         merged_data[i, :, 2] = temp  # 温度（所有节点相同）
#         merged_data[i, :, 3] = wind  # 风速（所有节点相同）
#         merged_data[i, :, 4] = np.argmax(wthr)  # 天气类型（所有节点相同）
#
# # 保存为 .npz 文件
# np.savez('TaxiBJ.npz', data=merged_data)
#
# print("数据融合完成并保存为 TaxiBJ.npz")
# print("转换后的数据形状:", merged_data.shape)  # 应该输出 (时间步数, 1024, 5)
#
# print(merged_data[:5])

# import h5py
# import numpy as np
# from datetime import datetime, timedelta
#
# # 加载流量数据
# flow_file = 'BJ16_M32x32_T30_InOut.h5'
# with h5py.File(flow_file, 'r') as f:
#     flow_dates = [d.decode('utf-8') for d in f['date'][:]]  # 转换为字符串列表
#     flow_data = f['data'][:]  # 形状: (num_timeslots, 2, 32, 32)
#
# # 解析流量日期并过滤无效数据
# valid_flow_datetimes = []
# valid_flow_data = []
# for date_str in flow_dates:
#     try:
#         dt = datetime.strptime(date_str, '%Y%m%d%H')
#         valid_flow_datetimes.append(dt)
#         valid_flow_data.append(flow_data[len(valid_flow_datetimes) - 1])
#     except ValueError:
#         print(f"跳过无效日期: {date_str}")
#
# # 加载天气数据
# meteo_file = 'BJ_Meteorology.h5'
# with h5py.File(meteo_file, 'r') as f:
#     meteo_dates = [d.decode('utf-8') for d in f['date'][:]]
#     temperature = f['Temperature'][:]
#     wind_speed = f['WindSpeed'][:]
#     weather = f['Weather'][:]
#
# # 解析天气日期并对齐到30分钟间隔
# meteo_datetimes = []
# for date_str in meteo_dates:
#     dt = datetime.strptime(date_str, '%Y%m%d%H%M')
#     # 对齐到最近的30分钟（00或30分）
#     minute = dt.minute // 30 * 30
#     aligned_dt = dt.replace(minute=minute, second=0, microsecond=0)
#     meteo_datetimes.append(aligned_dt)
#
# # 创建天气字典（以对齐后的时间为键）
# meteo_dict = {}
# for dt, temp, wind, wthr in zip(meteo_datetimes, temperature, wind_speed, weather):
#     meteo_dict[dt] = (temp, wind, np.argmax(wthr))
#
# # 合并数据
# num_timeslots = len(valid_flow_datetimes)
# num_nodes = 32 * 32  # 展平为1024个节点
# num_features = 5  # [in, out, temp, wind, weather]
#
# merged_data = np.zeros((num_timeslots, num_nodes, num_features))
#
# for i, flow_dt in enumerate(valid_flow_datetimes):
#     # 获取流量数据（2, 32, 32）
#     inflow = valid_flow_data[i][0].flatten()  # 展平为1024
#     outflow = valid_flow_data[i][1].flatten()
#
#     # 获取天气数据
#     meteo_vals = meteo_dict.get(flow_dt, (np.nan, np.nan, -1))
#
#     # 合并特征
#     merged_data[i, :, 0] = inflow
#     merged_data[i, :, 1] = outflow
#     merged_data[i, :, 2] = meteo_vals[0]  # 温度
#     merged_data[i, :, 3] = meteo_vals[1]  # 风速
#     merged_data[i, :, 4] = meteo_vals[2]  # 天气类型
#
# # 处理缺失值（可根据需求填充）
# if np.isnan(merged_data).any():
#     print("警告：存在缺失数据，建议进行插值处理")
#
# # 保存为npz文件
# np.savez('TaxiBJ.npz', data=merged_data)
# print("转换完成！最终数据形状:", merged_data.shape)
#
# print(merged_data[:5])


import h5py
import numpy as np
from datetime import datetime

# Load BJ16 data
with h5py.File('BJ16_M32x32_T30_InOut.h5', 'r') as f_bj16:
    dates_bj16 = [d.decode('utf-8') for d in f_bj16['date'][:]]
    data_bj16 = f_bj16['data'][:]  # Shape: (N, 2, 32, 32)

# Load meteorological data
with h5py.File('BJ_Meteorology.h5', 'r') as f_meteo:
    dates_meteo = [d.decode('utf-8') for d in f_meteo['date'][:]]
    temperature = f_meteo['Temperature'][:]
    wind_speed = f_meteo['WindSpeed'][:]
    weather = f_meteo['Weather'][:]  # Shape: (M, 17)


# Convert date strings to datetime objects
def parse_date(date_str):
    return datetime.strptime(date_str, '%Y%m%d%H%M')


dt_bj16 = [parse_date(d) for d in dates_bj16]
dt_meteo = [parse_date(d) for d in dates_meteo]

# Find common timesteps using two-pointer technique
i, j = 0, 0
common_indices = []
while i < len(dt_bj16) and j < len(dt_meteo):
    if dt_bj16[i] == dt_meteo[j]:
        common_indices.append((i, j))
        i += 1
        j += 1
    elif dt_bj16[i] < dt_meteo[j]:
        i += 1
    else:
        j += 1

# Initialize output array
num_timesteps = len(common_indices)
num_nodes = 32 * 32
num_features = 5  # [in, out, temp, wind, weather]
final_data = np.zeros((num_timesteps, num_nodes, num_features), dtype=np.float32)

# Merge features
for step, (idx_bj, idx_mt) in enumerate(common_indices):
    # Extract flow data
    inflow = data_bj16[idx_bj, 0].flatten()  # Shape: (1024,)
    outflow = data_bj16[idx_bj, 1].flatten()  # Shape: (1024,)

    # Extract weather features
    temp = temperature[idx_mt]
    wind = wind_speed[idx_mt]
    weather_code = np.argmax(weather[idx_mt])

    # Create feature matrix
    final_data[step, :, 0] = inflow
    final_data[step, :, 1] = outflow
    final_data[step, :, 2] = temp
    final_data[step, :, 3] = wind
    final_data[step, :, 4] = weather_code

# Save as compressed .npz file
np.savez_compressed('TaxiBJ.npz', data=final_data)

print(f"Conversion complete. Final shape: {final_data.shape}")

print(final_data[:5])