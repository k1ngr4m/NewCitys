import h5py
import numpy as np

# 打开 HDF5 文件
with h5py.File('BJ16_M32x32_T30_InOut.h5', 'r') as f:
    # 查看所有子集名称
    print("文件中的子集:", list(f.keys()))  # 预期输出: ['date', 'data']

    # 读取时间戳信息
    dates = f['date'][:]
    print("\n===== date =====")
    print("时间片数量:", dates.shape[0])
    print("时间戳格式示例:", dates[0].decode())  # 示例: '201511010000'（2015-11-01 00:00）

    # 读取流量数据
    data = f['data'][:]
    print("\n===== data =====")
    print("数据形状:", data.shape)  # 预期: (7220, 2, 32, 32)
    print("第一个时间片的流入矩阵（部分）:", data[0, 0, :5, :5])
    print("第一个时间片的流出矩阵（部分）:", data[0, 1, :5, :5])
    print("数据范围验证（min/max）:", np.min(data), np.max(data))  # 预期: 0.0, 1250.0

# 输出示例：
# 文件中的子集: ['date', 'data']
#
# ===== date =====
# 时间片数量: 7220
# 时间戳格式示例: 201511010000
#
# ===== data =====
# 数据形状: (7220, 2, 32, 32)
# 第一个时间片的流入矩阵（部分）: [[10.  0.  3.  0. 5.] ... ]
# 第一个时间片的流出矩阵（部分）: [[ 5.  8.  0. 12. 2.] ... ]
# 数据范围验证（min/max）: 0.0 1250.0

# 检查时间片数量是否与气象数据一致
with h5py.File('BJ_Meteorology.h5', 'r') as f_meteo:
    meteo_dates = f_meteo['date'][:]
    print("\n时间片对齐验证:", np.array_equal(dates, meteo_dates))  # 预期: True

missing_mask = np.isnan(data)
print("\n缺失值比例:", np.sum(missing_mask) / data.size)  # 预期: ~7.2%