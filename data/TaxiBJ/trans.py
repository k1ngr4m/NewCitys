import h5py
import numpy as np

# 读取HDF5文件
with h5py.File('BJ16_M32x32_T30_InOut.h5', 'r') as f:
    data = f['data'][:]  # 提取四维数据，形状为 (T, 2, 32, 32)

# 维度转换：将 (T, 2, 32, 32) 转换为 (T, 32*32, 2)
T = data.shape[0]
nodes = 32 * 32
reshaped_data = data.transpose(0, 2, 3, 1).reshape(T, nodes, 2)

# 保存为.npz文件
np.savez("TaxiBJ.npz", data=reshaped_data)

# 验证输出形状
print("转换后的数据形状:", reshaped_data.shape)