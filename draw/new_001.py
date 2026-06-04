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

# === 1. 数据与配置 ===
# 数据标签 (X轴)
models_xaxis = ['SA-MGSTFN', 'RemTimeGate', 'RemSemantics', 'RemDecoupling']
# 图例标签 (用于独立图例文件，无换行符)
models_legend = ['SA-MGSTFN', 'RemTimeGate', 'RemSemantics', 'RemDecoupling']

# 实验数据
mae_data = [17.88, 18.25, 18.85, 19.15]
rmse_data = [28.65, 29.32, 30.15, 31.42]
mape_data = [11.95, 12.28, 12.75, 12.98]

# 颜色定义 (蓝, 浅红, 绿, 橘黄)
custom_colors = ['#1f77b4', '#ff6666', '#2ca02c', '#ff7f0e']


# === 2. 函数：绘制不带图例的数据图 ===
def plot_metric_no_legend(metric_name, data, ylim_range, filename):
    plt.figure(figsize=(6, 5))

    # 绘制柱状图
    bars = plt.bar(models_xaxis, data, width=0.6, color=custom_colors, edgecolor='black', alpha=0.9)

    # 样式设置
    plt.title(f'{metric_name} 消融实验分析', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(fontsize=11)
    plt.ylim(ylim_range)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

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


# === 4. 执行生成 ===

# (A) 生成独立图例
plot_standalone_legend(models_legend, custom_colors, 'data/v2/Legend_Standalone.png')

# (B) 生成数据图
plot_metric_no_legend('MAE', mae_data, (16, 20), 'data/v2/MAE_ablation_clean.png')
plot_metric_no_legend('RMSE', rmse_data, (27, 33), 'data/v2/RMSE_ablation_clean.png')
plot_metric_no_legend('MAPE (%)', mape_data, (11, 14), 'data/v2/MAPE_ablation_clean.png')
