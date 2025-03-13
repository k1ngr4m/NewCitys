import h5py
import numpy as np

# 打开HDF5文件
with h5py.File('input.h5', 'r') as f:
    for key in f.keys():
        # 检查是否为数据集
        if isinstance(f[key], h5py.Dataset):
            data = f[key][:]  # 读取数据到NumPy数组
            np.save(f'{key}.npy', data)  # 保存为NPY文件