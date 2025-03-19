import numpy as np


def create_grid_adjacency_matrix(grid_size=32):
    """创建基于四邻域连接的网格邻接矩阵"""
    total_nodes = grid_size * grid_size
    adj = np.zeros((total_nodes, total_nodes), dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            current_node = i * grid_size + j
            # 上
            if i > 0:
                neighbor = (i - 1) * grid_size + j
                adj[current_node, neighbor] = 1.0
            # 下
            if i < grid_size - 1:
                neighbor = (i + 1) * grid_size + j
                adj[current_node, neighbor] = 1.0
            # 左
            if j > 0:
                neighbor = i * grid_size + (j - 1)
                adj[current_node, neighbor] = 1.0
            # 右
            if j < grid_size - 1:
                neighbor = i * grid_size + (j + 1)
                adj[current_node, neighbor] = 1.0
    return adj


# 生成32x32网格的邻接矩阵
adj_matrix = create_grid_adjacency_matrix(grid_size=32)

# 保存为npy文件
np.save('TaxiBJ_rn_adj.npy', adj_matrix)

print(f"邻接矩阵形状: {adj_matrix.shape}")
print("已保存为 TaxiBJ_rn_adj.npy")

# 加载验证
adj = np.load('TaxiBJ_rn_adj.npy')
print("中心节点(16*32+16=528)的邻居数:", adj[528].sum())  # 预期输出4.0
print("角落节点(0,0)的邻居数:", adj[0].sum())          # 预期输出2.0