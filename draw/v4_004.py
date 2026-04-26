import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 1. 设置中文字体 (为了正确显示中文)
# 如果你的环境没有 SimHei，可以尝试更改为 'Microsoft YaHei' (Windows) 或 'Arial Unicode MS' (Mac)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 2. 模拟数据 (构建一个 10x10 的矩阵来模仿图中的热度分布)
# 初始化全低值矩阵 (深蓝色背景)
data = np.random.rand(10, 10) * 0.15

# 模拟高亮区域 (Node 2 和 Node 8 的交叉点)
# 注意：Python索引从0开始，Node 2 是索引1，Node 8 是索引7
target_row, target_col = 1, 7

# 设置热点中心 (深红色)
data[target_row, target_col] = 1.0

# 设置热点周围的辐射 (黄色/绿色/浅蓝)
# 横向扩散
data[target_row, target_col-1] = 0.7  # Node 2, Node 7
data[target_row, target_col-2] = 0.4  # Node 2, Node 6
data[target_row, target_col+1] = 0.4  # Node 2, Node 9
data[0] = (0.0, 0.2, 0.1, 0.15, 0.17, 0.25, 0.2, 0.4, 0.2, 0.1)
data[1] = (0.2, 0.0, 0.2, 0.25, 0.3, 0.4, 0.55, 1.0, 0.4, 0.2)
data[3] = (0.1, 0.2, 0.0, 0.15, 0.16, 0.25, 0.2, 0.75, 0.2, 0.1)

# 纵向扩散
data[target_row+1, target_col] = 0.75  # Node 3, Node 8
data[target_row+2, target_col] = 0.45  # Node 4, Node 8
data[target_row-1, target_col] = 0.45  # Node 1, Node 8

np.fill_diagonal(data, 0)
# 3. 绘制热力图
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

# 使用 'jet' 颜色映射来模仿原图的 蓝-青-黄-红 风格
im = ax.imshow(data, cmap='jet', aspect='auto', interpolation='nearest')

# 4. 设置坐标轴
# Y轴: Node 1 到 Node 10
ax.set_yticks(np.arange(10))
ax.set_yticklabels([f'Node {i+1}' for i in range(10)], fontsize=10)
ax.set_ylabel('Node', fontsize=12)

# X轴: 根据原图显示特定的标签 (Node 1, 4, 6, 8, 10)
xticks_indices = [0, 3, 5, 7, 9]
xticks_labels = ['Node 1', 'Node 4', 'Node 6', 'Node 8', 'Node 10']
ax.set_xticks(xticks_indices)
ax.set_xticklabels(xticks_labels, fontsize=10)
ax.set_xlabel('Node index', fontsize=12)
# 标题
ax.set_title('(d) 语义注意力热力图 ($A_{sem}$) (Semantic Attention Heatmap)', fontsize=13, pad=15)
# 5. 添加颜色条 (Colorbar)
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=10)


# 6. 添加标注 (圆圈和箭头文字)
# 添加圆圈高亮关键点
circle = patches.Circle((target_col, target_row), radius=0.6, linewidth=1.5, edgecolor='black', facecolor='none')
ax.add_patch(circle)

# 添加箭头和说明文字
text_str = (
    "高语义关联\n"
    "Node 2 & Node 8\n"
    "(e.g., 购物中心,\n5km apart)"
)

# xy是箭头尖端坐标，xytext是文字坐标
ax.annotate(
    text_str,
    xy=(target_col + 0.6, target_row),      # 箭头指向圆圈右侧
    xytext=(target_col + 3.5, target_row + 3),  # 文字放在右下方
    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
    fontsize=10,
    ha='left',
    va='top'
)

# 7. 调整布局并保存
plt.tight_layout()
save_path = 'semantic_heatmap_reproduced.png'
plt.savefig(save_path)
print(f"图片已保存为: {save_path}")

plt.show()