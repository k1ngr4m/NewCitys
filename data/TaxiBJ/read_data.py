import h5py
import numpy as np
# 打开 HDF5 文件
with h5py.File('BJ_Meteorology.h5', 'r') as f:
    # 查看所有子集名称
    print("文件中的子集:", list(f.keys()))

    # 读取并打印各子集信息
    print("\n===== date =====")
    date = f['date'][:]
    print("形状:", date.shape)
    print("示例数据（前5项）:", date[:-5])

    print("\n===== Temperature =====")
    temperature = f['Temperature'][:]
    print("形状:", temperature.shape)
    print("示例数据（前5项）:", temperature[:5])

    print("\n===== WindSpeed =====")
    windspeed = f['WindSpeed'][:]
    print("形状:", windspeed.shape)
    print("示例数据（前5项）:", windspeed[:5])

    print("\n===== Weather =====")
    weather = f['Weather'][:]
    print("形状:", weather.shape)
    print("示例数据（第1个样本的天气类型索引）:", np.argmax(weather[0]))

# 输出结果示例：
# 文件中的子集: ['Weather', 'WindSpeed', 'date', 'Temperature']
#
# ===== date =====
# 形状: (7220,)
# 示例数据（前5项）: [b'201511010000' b'201511010030' b'201511010100' b'201511010130' b'201511010200']
#
# ===== Temperature =====
# 形状: (7220