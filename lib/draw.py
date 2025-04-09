import matplotlib.pyplot as plt
import numpy as np

# 定义组件名称常量，将 RemWeather 放在第二个位置
COMPONENTS = ['NewCity', 'RemWeather', 'RemTime', 'RemLPE']

# 调整 MAE 数据顺序
mae_data = [1.9, 2.13, 2.28, 2.27]
# 调整 MAPE 数据顺序
mape_data = [39.98, 45.88, 46.33, 45.01]
# 调整 RMSE 数据顺序
rmse_data = [4.3, 5.26, 5.58, 5.53]

# 定义颜色列表
colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']

bar_width = 0.8
r = np.arange(len(COMPONENTS))

# 设置图形大小
fig_size = (8, 6)

# 绘制 MAE 的柱状图
plt.figure(figsize=fig_size)
bars_mae = plt.bar(r, mae_data, width=bar_width, color=colors)
plt.xticks(r, COMPONENTS)
plt.xlabel('Components')
plt.ylabel('MAE Value')
plt.title('MAE Values of Different Components')
plt.tight_layout()
plt.show()
plt.savefig('mae_plot.png')
plt.close()

# 绘制 MAPE 的柱状图
plt.figure(figsize=fig_size)
bars_mape = plt.bar(r, mape_data, width=bar_width, color=colors)
plt.xticks(r, COMPONENTS)
plt.xlabel('Components')
plt.ylabel('MAPE Value')
plt.title('MAPE Values of Different Components')
plt.tight_layout()
plt.savefig('mape_plot.png')
plt.close()

# 绘制 RMSE 的柱状图
plt.figure(figsize=fig_size)
bars_rmse = plt.bar(r, rmse_data, width=bar_width, color=colors)
plt.xticks(r, COMPONENTS)
plt.xlabel('Components')
plt.ylabel('RMSE Value')
plt.title('RMSE Values of Different Components')
plt.tight_layout()
plt.savefig('rmse_plot.png')
plt.close()

# 创建图例
handles = []
for bar, component in zip(bars_mae, COMPONENTS):
    patch = plt.Rectangle((0, 0), 1, 1, fc=bar.get_facecolor())
    handles.append(patch)

# 绘制单独的图例图片
fig = plt.figure(figsize=(fig_size[0], 1))
fig.legend(handles, COMPONENTS, loc='center', ncol=len(COMPONENTS))
plt.axis('off')
plt.tight_layout()
plt.savefig('legend_plot.png')
plt.close()
