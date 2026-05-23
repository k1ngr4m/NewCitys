import os

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import platform

# --- 1. 设置中文字体 ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 生成数据 ---
x = np.linspace(0, 180, 500)
mu = 90
sigma = 28
height = 1.03
y = height * np.exp(-((x - mu)**2) / (2 * sigma**2))

# --- 3. 绘图 ---
fig, ax = plt.subplots(figsize=(8, 4))

# 定义颜色 (你可以修改这里的颜色代码)
line_color = '#1f77b4'  # 一种好看的蓝色
fill_color = '#a6cee3'  # 较浅的蓝色，用于填充

# 1. 绘制曲线 (修改 color 参数)
ax.plot(x, y, color=line_color, linewidth=2)

# 2. (可选) 添加填充颜色，让图表更生动
# alpha=0.2 表示 20% 的不透明度，很淡
ax.fill_between(x, y, color=fill_color, alpha=0.5)

# === 修改部分结束 ===

# --- 4. 设置坐标轴 ---
ax.set_xlabel("Time", fontsize=12)
ax.set_xlim(0, 180)

ax.set_ylabel(r"$Z_{inv}$ Value", fontsize=12)
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


tick_locs = [0, 30, 60, 90, 120, 150, 180]
tick_labels = ['17:00', '17:30', '18:00', '18:30', '19:00', '19:30', '20:00']
ax.set_xticks(tick_locs)
ax.set_xticklabels(tick_labels)

ax.grid(True, linestyle='--', alpha=0.6) # 添加虚线网格

# --- 6. 显示与保存 ---
plt.tight_layout()

# 确保保存目录存在
save_dir = 'data/v4'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, '002_colored.png')

# 注意：plt.savefig 应该在 plt.show() 之前调用，否则保存的图片可能是空白的
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"图片已保存至: {save_path}")