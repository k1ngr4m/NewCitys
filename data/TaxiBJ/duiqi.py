import h5py
from datetime import datetime
import numpy as np

# 定义所有年份的数据文件路径
year_files = [
    'BJ13_M32x32_T30_InOut.h5',
    'BJ14_M32x32_T30_InOut.h5',
    'BJ15_M32x32_T30_InOut.h5',
    'BJ16_M32x32_T30_InOut.h5'
]

# 存储所有时间片及其对应的数据和来源
all_dates = {}  # 格式: {datetime_obj: (data_array, file_index)}

for file_idx, file in enumerate(year_files):
    with h5py.File(file, 'r') as f:
        # 读取时间戳和流量数据
        dates = [datetime.strptime(d.decode(), "%Y%m%d%H%M") for d in f['date'][:]]
        data = f['data'][:]  # 形状: (T, 2, 32, 32)

        # 按时间戳存储数据，覆盖重复项（保留最新文件的数据）
        for i, d in enumerate(dates):
            all_dates[d] = (data[i], file_idx)

# 按时间排序
sorted_dates = sorted(all_dates.keys())

# 初始化合并后的数据数组
num_timeslots = len(sorted_dates)
merged_data = np.zeros((num_timeslots, 2, 32, 32), dtype=np.float32)

# 填充数据（自动处理重复时间片，保留最后出现的文件数据）
for i, d in enumerate(sorted_dates):
    merged_data[i] = all_dates[d][0]

print("合并后的数据形状:", merged_data.shape)  # 预期: (Total_TimeSlots, 2, 32, 32)

# 检查时间间隔是否为30分钟
for i in range(1, len(sorted_dates)):
    delta = sorted_dates[i] - sorted_dates[i-1]
    if delta.total_seconds() != 1800:
        print(f"时间片不连续: {sorted_dates[i-1]} -> {sorted_dates[i]}")

