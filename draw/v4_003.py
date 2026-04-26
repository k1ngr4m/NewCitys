import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. 设置中文字体 (根据你的系统调整) ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 生成数据 ---
time_minutes = np.arange(0, 181, 1)
z_spe = np.zeros_like(time_minutes, dtype=float)

start_idx = 90
duration = 7
end_idx = start_idx + duration

# 设置脉冲值
z_spe[start_idx:end_idx] = np.linspace(1.0, 0.99, duration)
# 下降沿后的微小波动
if end_idx + 5 < len(z_spe):
    z_spe[end_idx:end_idx+3] = [0.2, 0.1, 0.05]

# --- 3. 绘图配置 ---
fig, ax = plt.subplots(figsize=(8, 4))

# 定义颜色变量
line_color = '#1f77b4'  # 一种好看的蓝色
fill_color = '#a6cee3'  # 较浅的蓝色，用于填充

# --- 【修改点 1】绘制线条 ---
# 将 color 改为 line_color，并稍微增加线宽
ax.plot(time_minutes, z_spe, color=line_color, linewidth=2)

# --- 【修改点 2】添加填充色 ---
# 使用 fill_between 在曲线和x轴之间添加半透明填充
ax.fill_between(time_minutes, 0, z_spe, color=fill_color, alpha=0.5)

# --- 4. 设置坐标轴 ---
ax.set_xlabel('Time', fontsize=12)
ax.set_xlim(0, 180)

ax.set_ylabel(r'$Z_{spe}$ Value', fontsize=12) # y轴标签也用蓝色点缀
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


xticks_loc = [0, 30, 60, 90, 120, 150, 180]
xticks_labels = ['17:00', '17:30', '18:00', '18:30', '19:00', '19:30', '20:00']
ax.set_xticks(xticks_loc)
ax.set_xticklabels(xticks_labels)

# --- 【修改点 3】添加网格和图例 ---
ax.grid(True, linestyle='--', alpha=0.6) # 添加虚线网格
# ax.legend(loc='upper right', frameon=True, shadow=True) # 添加图例

# --- 5. 添加标注和标题 (保持注释状态) ---
# ax.set_title(r'(c) 突发干扰特征 ($Z_{spe}$)', fontsize=13, pad=10)
# arrow_x, arrow_y = 98, 0.7
# text_x, text_y = 115, 0.8
# ax.annotate('突发干扰激活', xy=(arrow_x, arrow_y), xytext=(text_x, text_y),
#             arrowprops=dict(facecolor=line_color, edgecolor=line_color, arrowstyle='->', connectionstyle="arc3"),
#             fontsize=10, ha='left', va='center', color=line_color)

# --- 6. 显示与保存 ---
plt.tight_layout()

# 确保保存目录存在
save_dir = 'data/v4'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, '003_colored.png')

# 注意：plt.savefig 应该在 plt.show() 之前调用，否则保存的图片可能是空白的
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"图片已保存至: {save_path}")