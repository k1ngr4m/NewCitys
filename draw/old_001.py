import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager


def setup_chinese_font():
    """Auto-select an available CJK font to avoid tofu boxes."""
    candidates = [
        'PingFang SC',
        'Hiragino Sans GB',
        'STHeiti',
        'Heiti SC',
        'Songti SC',
        'Arial Unicode MS',
        'Microsoft YaHei',
        'SimHei',
        'Noto Sans CJK SC',
        'WenQuanYi Zen Hei',
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            break
    plt.rcParams['axes.unicode_minus'] = False


setup_chinese_font()

# === 1. 数据准备 (基于提供的图片提取) ===
# 简化标签名称，方便图表展示
models_xaxis = ['MSTDFN', 'RemWeather', 'RemTime', 'RemLPE']
models_legend = ['MSTDFN', 'RemWeather', 'RemTime', 'RemLPE']

# 提取的数值
mae_data = [1.9, 2.13, 2.28, 2.27]
rmse_data = [4.3, 5.26, 5.58, 5.53]
mape_data = [39.98, 45.88, 46.33, 45.01]  # 单位: %

custom_colors = ['#1f77b4', '#ff6666', '#2ca02c', '#ff7f0e']

# === 2. 定义通用绘图函数 ===
def plot_metric_no_legend(metric_name, data, ylim_range, filename):
    """
    绘制单个指标的柱状图
    """
    plt.figure(figsize=(6, 5))

    # 绘制柱子
    bars = plt.bar(models_xaxis, data, width=0.6, color=custom_colors, edgecolor='black', alpha=0.9)

    # 标题与轴标签
    plt.title(f'{metric_name} 消融实验分析', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(fontsize=11)
    plt.ylim(ylim_range)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # 设置Y轴范围 (关键：截断Y轴以突显差异)
    plt.ylim(ylim_range)


    plt.tight_layout()
    print(f"Saving Chart: {filename}...")
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()  # 关闭画布释放内存



# === 3. 函数：绘制单独的图例 ===
def plot_standalone_legend(labels, colors, filename):
    # 创建一个适合放图例的画布 (宽8, 高1)
    fig = plt.figure(figsize=(8, 1))

    # 创建一个隐藏坐标轴的子图
    ax = fig.add_subplot(111)
    ax.axis('off')

    # 创建图例的句柄 (Handles)
    # 使用 mpatches.Patch 创建对应颜色的色块
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]

    # 绘制图例
    # loc='center': 居中
    # ncol=4: 分4列横向排列 (如果想竖排改成 ncol=1)
    # frameon=False: 去掉图例的边框，更适合嵌入论文
    legend = ax.legend(handles=patches, loc='center', ncol=4, fontsize=12, frameon=False)

    # 保存时使用 bbox_inches='tight' 裁剪掉多余空白
    print(f"Saving Legend: {filename}...")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


# === 3. 执行绘图 (根据数据调整了Y轴范围) ===

plot_standalone_legend(models_legend, custom_colors, 'data/v3/Legend_Standalone.png')

# (B) 生成数据图
plot_metric_no_legend('MAE', mae_data, (1.5, 2.5), 'data/v3/MAE_ablation_clean.png')
plot_metric_no_legend('RMSE', rmse_data, (3.5, 6.0), 'data/v3/RMSE_ablation_clean.png')
plot_metric_no_legend('MAPE (%)', mape_data, (35, 50), 'data/v3/MAPE_ablation_clean.png')
