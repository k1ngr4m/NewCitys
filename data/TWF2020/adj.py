import re

import pandas as pd
import numpy as np

# 读取 CSV 文件
df = pd.read_csv('TWF2020.csv')

# 清理并转换数据类型（如果需要）
def clean_temperature(temp_str):
    if isinstance(temp_str, str):
        cleaned = re.sub(r'[^\d.]+', '', temp_str)
        return float(cleaned)
    else:
        return temp_str

if df['Temperature'].dtype == object:
    df['Temperature'] = df['Temperature'].apply(clean_temperature)

df['Flow'] = df['Flow'].astype(float)

# 假设我们基于 'Flow' 列来构建邻接矩阵
flow_data = df['Flow'].values.reshape(-1, 1)

# 计算皮尔逊相关系数矩阵
corr_matrix = np.corrcoef(flow_data.T)

# 如果想要基于绝对相关系数，可以如下操作
adj_matrix = np.abs(corr_matrix)

# 打印邻接矩阵的形状以确认大小
print("Shape of the adjacency matrix:", adj_matrix.shape)

# 保存为 .npy 或 .npz 文件
np.save('TWF2020_rn_adj.npy', adj_matrix)
# 或者
np.savez('adjacency_matrix.npz', data=adj_matrix)

print("Adjacency matrix saved.")