import h5py

# 打开 HDF5 文件
with h5py.File('BJ_Meteorology.h5', 'r') as f:
    # 获取各个子集的数据
    dates = f['date'][:]
    temperatures = f['Temperature'][:]
    wind_speeds = f['WindSpeed'][:]
    weathers = f['Weather'][:]

    print(f"Dates shape: {dates.shape}")
    print(f"Temperatures shape: {temperatures.shape}")
    print(f"Wind speeds shape: {wind_speeds.shape}")
    print(f"Weathers shape: {weathers.shape}")