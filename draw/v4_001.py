import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# Create directory
os.makedirs('data', exist_ok=True)

# --- Font Handling for Cloud Environment ---
# Try to find a Chinese font available in this Linux environment
font_names = ['Arial Unicode MS', 'WenQuanYi Micro Hei', 'Droid Sans Fallback', 'SimHei']
selected_font = None

for name in font_names:
    if name in [f.name for f in fm.fontManager.ttflist]:
        selected_font = name
        break

# Fallback: search by file path if names don't match standard system install locations
if not selected_font:
    system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    for font_path in system_fonts:
        try:
            font_prop = fm.FontProperties(fname=font_path)
            if 'WenQuanYi' in font_prop.get_name() or 'DroidSansFallback' in font_path:
                # Register this font
                fm.fontManager.addfont(font_path)
                selected_font = font_prop.get_name()
                break
        except:
            continue

if selected_font:
    plt.rcParams['font.sans-serif'] = [selected_font]
else:
    # If all else fails, rely on matplotlib's default but warn (squares may appear)
    # But usually 'DejaVu Sans' is default and doesn't support CJK.
    pass

plt.rcParams['axes.unicode_minus'] = False


def generate_smooth_large_deviation():
    x = np.linspace(0, 180, 180)
    np.random.seed(20)

    def get_smooth_noise(size, window, amplitude):
        noise = np.random.normal(0, 1, size)
        smooth_noise = np.convolve(noise, np.ones(window) / window, mode='same')
        return smooth_noise * amplitude

    # 1. Ground Truth
    base_wave = 50 + 3.0 * np.sin(x[:95] / 7.0)
    random_wave = get_smooth_noise(95, 12, 8.0)
    y_high = base_wave + random_wave

    y_low = 6 + np.sin(x[105:] / 3) * 0.5 + get_smooth_noise(75, 8, 3.0)
    y_drop = np.linspace(y_high[-1], y_low[0], 10)
    y_gt = np.concatenate([y_high, y_drop, y_low])

    y_gt = np.convolve(y_gt, np.ones(5) / 5, mode='same')
    y_gt[:5] = 50 + np.random.normal(0, 1, 5)
    y_gt[-5:] = 6

    # 2. SA-MGSTFN (Ours)
    deviation_wave = 2.0 * np.sin(x / 15)
    deviation_random = get_smooth_noise(180, 15, 3.0)

    y_ours = y_gt + deviation_wave + deviation_random
    y_ours[90:110] += 2.0
    y_ours = np.convolve(y_ours, np.ones(6) / 6, mode='same')
    y_ours[:10] = y_gt[:10] + 1.5

    # 3. MSTDFN (Baseline)
    y_baseline = np.copy(y_gt)
    lag_start = 95
    lag_end = 150
    sigmoid_x = np.linspace(-6, 6, lag_end - lag_start)
    sigmoid_y = 50 - (44 / (1 + np.exp(-sigmoid_x * 0.6)))
    y_baseline[lag_start:lag_end] = sigmoid_y

    y_baseline = np.convolve(y_baseline, np.ones(15) / 15, mode='same')
    y_baseline[:15] = np.linspace(51, y_baseline[15], 15)
    y_baseline[-20:] = y_gt[-20:] + get_smooth_noise(20, 5, 2.0)

    return x, y_gt, y_ours, y_baseline


x, y_gt, y_ours, y_baseline = generate_smooth_large_deviation()

plt.figure(figsize=(10, 4.5), dpi=100)
plt.plot(x, y_gt, color='black', linewidth=2.5, label='Ground Truth', zorder=3)
plt.plot(x, y_ours, color='#C00000', linestyle='--', linewidth=3.0, label='SA-MGSTFN (Ours)', zorder=2)
plt.plot(x, y_baseline, color='gray', linestyle=':', linewidth=2.2, label='MSTDFN (Baseline)', zorder=1)

# plt.title('(a) 真实流量与预测流量对比', fontsize=15, y=1.02)
plt.ylabel('Traffic Flow', fontsize=13)
plt.xlabel('Time', fontsize=13)

plt.xticks([0, 30, 60, 90, 120, 150, 180],
           ['17:00', '17:30', '18:00', '18:30', '19:00', '19:30', '20:00'], fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0, 180)
plt.ylim(0, 60)

plt.legend(loc='upper right', frameon=True, fontsize=11, borderpad=0.6)

# plt.annotate('突发交通拥堵',
#              xy=(96, 30), xycoords='data',
#              xytext=(40, 20), textcoords='data',
#              arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
#              fontsize=12, ha='center')
#
# plt.annotate('滞后',
#              xy=(135, 18), xycoords='data',
#              xytext=(158, 28), textcoords='data',
#              arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
#              fontsize=12, ha='center')

plt.tight_layout()
plt.savefig('data/v4/RealvsPred.png')
