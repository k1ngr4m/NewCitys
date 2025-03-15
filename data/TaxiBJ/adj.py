import numpy as np


def generate_grid_adjacency(grid_size=32):
    num_nodes = grid_size * grid_size
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # 定义四个邻域方向（上、下、左、右）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(grid_size):
        for j in range(grid_size):
            current = i * grid_size + j
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    neighbor = ni * grid_size + nj
                    adj[current, neighbor] = 1.0
    # 添加自连接
    adj += np.eye(num_nodes)
    return adj


# 生成并保存邻接矩阵
adj = generate_grid_adjacency(grid_size=32)
np.save('TaxiBJ_rn_adj.npy', adj)