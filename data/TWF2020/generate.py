import numpy as np
import pandas as pd
import re

def clean_temperature(temp_str):
    """Remove non-ASCII characters and units, then convert to float."""
    if isinstance(temp_str, str):
        # 使用正则表达式去除所有非数字字符和不可见字符，除了小数点
        cleaned = re.sub(r'[^\d.]+', '', temp_str)
        return float(cleaned)
    else:
        return temp_str
def change():
    # 读取 CSV 文件
    csv_file_path = 'TWF2020.csv'
    df = pd.read_csv(csv_file_path)

    # 提取所需的特征列
    selected_columns = ['5 Minutes', 'Flow', 'Temperature']
    data = df[selected_columns]



    # 应用清理函数到 'Temperature' 列
    df['Temperature'] = df['Temperature'].apply(clean_temperature)

    features = df[['Flow', '(mph)', 'Temperature']].values  # 形状为 (n_samples, 3)

    three_dim_array = features.reshape(-1, 1, 3)

    # 保存为 .npz 文件
    np.savez_compressed('TWF2020.npz', data=three_dim_array)

    print(f"数据已保存到 TWF2020.npz")


def load_npz(name):
    # 加载NPZ文件
    with np.load(name + '.npz') as npz_data:
        # 列出所有数据集的键名
        keys = npz_data.files
        print("NPZ文件中的数据集键名:", keys)

        # 遍历所有数据集并打印信息
        for key in keys:
            dataset = npz_data[key]
            print(f"\n=== 数据集: {key} ===")
            print("数据类型:", dataset.dtype)
            print("数据形状:", dataset.shape)
            print("部分数据示例:\n", dataset[0] if dataset.ndim > 1 else dataset[:5])

if __name__ == '__main__':
    change()
    file_name = 'TWF2020'
    load_npz(file_name)