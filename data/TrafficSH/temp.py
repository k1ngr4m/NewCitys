import h5py
import numpy as np
import os

def h5_to_npz(h5_path, npz_path):
    with h5py.File(h5_path, 'r') as h5_file:
        data = {}
        # 遍历所有数据集并收集数据
        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                data[name] = np.array(obj)
        h5_file.visititems(collect_datasets)
    # 保存所有数据集到NPZ文件
    np.savez(npz_path, **data)

def h5_to_npy(h5_path, output_dir):
    with h5py.File(h5_path, 'r') as h5_file:
        def save_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                # 处理路径：去掉开头的斜杠并分割目录和文件名
                relative_path = name.lstrip('/')
                dir_path = os.path.join(output_dir, os.path.dirname(relative_path))
                os.makedirs(dir_path, exist_ok=True)  # 创建目录
                file_name = os.path.basename(relative_path) + '.npy'
                file_path = os.path.join(dir_path, file_name)
                np.save(file_path, np.array(obj))  # 保存数据
        h5_file.visititems(save_dataset)
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
def load_npy(name):
    # 加载NPY文件
    data = np.load(name + '.npy')

    # 查看内容和形状
    print("=== NPY文件内容 ===")
    print("数据类型:", data.dtype)  # 例如：float32、uint8等
    print("数据形状:", data.shape)  # 例如：(100, 200)表示100行200列的数组
    print("前5个元素示例:\n", data[0] if data.ndim > 1 else data[:5])  # 显示部分数据
if __name__ == '__main__':
    file_name = 'TrafficSH'
    # h5_to_npz(file_name + '.h5', file_name + '.npz')
    # h5_to_npy(file_name + '.h5', file_name)
    # data = np.load(file_name + '.npz')
    # print(data.files)  # 查看所有键名
    # dataset = data['group/dataset']  # 访问特定数据集

    # dataset = np.load('output_directory/group/dataset.npy')
    load_npz(file_name)

    file_names = 'TrafficSH_rn_adj'
    load_npy(file_names)