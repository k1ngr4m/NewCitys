import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# === 0. 设置中文字体（自动选择当前机器可用字体）===
def setup_chinese_font():
    candidates = [
        "PingFang SC",          # macOS
        "Hiragino Sans GB",     # macOS
        "Microsoft YaHei",      # Windows
        "SimHei",               # Windows/Linux
        "Noto Sans CJK SC",     # Linux/跨平台
        "Source Han Sans SC",   # Linux/跨平台
        "WenQuanYi Micro Hei",  # Linux
        "Arial Unicode MS",     # 部分 macOS/Office 环境
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    selected = next((name for name in candidates if name in available), None)

    if selected:
        # 同时设置 family 和 sans-serif，避免被默认 Arial 覆盖
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [selected, "Arial Unicode MS", "DejaVu Sans"]
        print(f"[INFO] 使用中文字体: {selected}")
    else:
        print("[WARN] 未找到可用中文字体，中文可能显示为方块。")

    plt.rcParams["axes.unicode_minus"] = False


# === 1. 模拟数据 (请替换为你实验中的真实数据) ===
# 假设时间段是 6:00 到 12:00，每10分钟一个点，共36个点
time_slots = np.arange(36)
x_labels = ['06:00', '07:00', '08:00', '09:00', '10:00', '11:00']

# 构造故事：早高峰正常应该上升，但8:00-9:30下雨，流量骤降
# Ground Truth: 真实值（受下雨影响，中间凹陷）
ground_truth = np.array([20, 25, 35, 45, 55, 65, 75, 80, 85, 90, 92, 95,  # 6:00-8:00 正常上升
                         80, 70, 65, 60, 58, 60, 65, 75, 85,           # 8:00-9:30 下雨，流量异常下降
                         90, 92, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30]) # 恢复

# RemWeather: 无天气模块（它不知道下雨，只按周期性规律预测，所以它是平滑的抛物线）
rem_weather = np.array([22, 26, 34, 46, 54, 66, 74, 82, 86, 91, 93, 96,
                        98, 100, 102, 103, 102, 100, 98, 95, 92,        # 这里预测偏高，产生巨大误差！
                        90, 88, 86, 84, 80, 76, 72, 68, 64, 58, 52, 46, 42, 36, 32])

# MSTDFN: 完整模型（感知到下雨，虽然有误差，但趋势跟上了）
mstdfn =      np.array([21, 24, 36, 44, 56, 64, 76, 79, 84, 89, 91, 93,
                        85, 75, 68, 64, 62, 63, 68, 77, 86,             # 成功预测到了下降趋势
                        89, 91, 89, 84, 79, 74, 69, 64, 61, 56, 51, 44, 41, 34, 31])

# === 2. 绘图 ===
plt.figure(figsize=(10, 5), dpi=300)

# 设置风格
plt.style.use('seaborn-v0_8-whitegrid')
# 注意：style.use 可能覆盖字体配置，所以字体设置必须放在它后面
setup_chinese_font()

# 画线
plt.plot(time_slots, ground_truth, color='black', linewidth=2.5, label='真实流量')
plt.plot(time_slots, rem_weather, color='red', linestyle='--', linewidth=2, label='RemWeather')
plt.plot(time_slots, mstdfn, color='#1f77b4', linewidth=2.5, marker='o', markersize=4, label='MSTDFN (Ours)')

# === 3. 关键：添加“下雨”背景区域 ===
# 假设下雨时间是第12个点到第21个点 (08:00 - 09:30)
plt.axvspan(12, 21, color='blue', alpha=0.15, label='降雨时段')
# 在背景上方添加文字说明
# plt.text(12.5, 105, 'Heavy Rain', ha='center', va='bottom', fontsize=12, color='blue', fontweight='bold')

# === 4. 细节调整 ===
# plt.title('Case Study: Traffic Flow Prediction under Heavy Rain (NYC-BIKE)', fontsize=14, pad=15)
plt.xlabel('时间', fontsize=12)
plt.ylabel('交通流量', fontsize=12)
plt.legend(loc='upper right', frameon=True, framealpha=0.9, shadow=True)

# 设置X轴刻度
plt.xticks(np.arange(0, 36, 6), x_labels)
plt.xlim(0, 35)
plt.ylim(0, 115) # 留出顶部空间写字

# # 标记误差区域（可选，用箭头指出差距）
# plt.annotate('Large Error', xy=(15, 102), xytext=(18, 110),
#              arrowprops=dict(facecolor='red', shrink=0.05),
#              fontsize=10, color='red')

plt.tight_layout()
plt.show()
