import h5py
import os
import numpy as np


def read_h5_file(file_path):
    """
    从.h5文件中读取数据并返回一个字典。

    Parameters:
    file_path (str): .h5文件的路径。

    Returns:
    data (dict): 包含从.h5文件中读取的数据的字典。
    """
    data = {}  # 创建一个空字典以存储数据

    try:
        with h5py.File(file_path, 'r') as h5file:
            # 遍历.h5文件中的所有数据集
            for dataset_name in h5file.keys():
                dataset = h5file[dataset_name]
                data[dataset_name] = dataset[:]  # 将数据集的内容存储在字典中
    except Exception as e:
        print(f"读取.h5文件时出错：{str(e)}")
        data = None

    return data


def save_h5_file(np_data=None, dataset_name=None, out_h5_file_path=""):
    """
    将ndarray 写为 一个 .h5文件

    Parameters:
    np_data (ndarray)： 要写入的数据
    dataset_name(str): 写入数据 要定义的数据集的名字
    out_h5_file_path(str): 输出h5文件路径(带文件名)

    用法：
        # a = torch.cat(self.patches_low, dim=0).numpy()
        # b = torch.cat(self.patches_high, dim=0).numpy()
        # from dataset.a_Helper.PC_utils import h5_utils
        # h5_utils.save_h5_file([a, b], ["poisson_256", "poisson_1024"], "sketchfab_256_1024_poisson.h5")
        # print("finish")
    """
    with h5py.File(out_h5_file_path, 'w') as hdf5_file:
        # 创建一个数据集并将数据写入其中，指定数据集的名称为 'data'
        for name, ndarray in zip(dataset_name, np_data):
            hdf5_file.create_dataset(name, data=ndarray)


def add_ndarray_to_h5_file(input_h5="", new_h5="", new_key=None, new_data=None):
    """
        在现有的.h5文件基础上 添加ndarray 创建新的h5

        Parameters:
        add_np_data (ndarray)： 要添加的数据
        dataset_name(str): 添加到已有(新)数据集的名字
        exist_h5_file(str): 已有h5文件路径(带文件名)
    """
    with h5py.File(input_h5, 'r') as input_file:
        with h5py.File(new_h5, 'w') as output_file:  # 创建一个新的 HDF5 文件
            for key in input_file.keys():
                # 将数据集复制到新的文件中，目标路径与原路径相同
                input_file.copy(key, output_file, name=key)
            if isinstance(new_key, list) and isinstance(new_data, list):
                for key, data in zip(new_key, new_data):
                    output_file.create_dataset(key, data=data)
            elif isinstance(new_key, str) and isinstance(new_data, np.ndarray):
                output_file.create_dataset(new_key, data=new_data)  # 将新的三维 ndarray 添加到新的 HDF5 文件中
            else:
                print("type error. The current type is: {} and {}".format(new_key.type(), new_data.type())
                      + "\n"
                      + "type should be str and ndarray, or list and list")




def h5_add_h5_file(data_h5="", data_h5_dataset_name="",
                   add_to_h5="", add_to_h5_dataset_name=""):
    """
    将ndarray 添加到现有的.h5文件

    Parameters:
    data_h5 (.h5)： 要添加的数据来源h5文件
    data_h5_dataset_name(str): 要添加的数据来源h5文件中的数据集名字
    add_to_h5(.h5): 要添加至目标的h5文件
    exist_h5_file(str): 要添加至目标的h5文件中的数据集名字
    """
    add_data = read_h5_file(data_h5)[data_h5_dataset_name]
    ndarray_add_h5_file(exist_h5_file=add_to_h5, add_np_data=add_data, dataset_name=add_to_h5_dataset_name)

    # from utils.my_utils import h5_add_h5_file
    #
    # data_h5 = "checkpoint/dense_channel_split_high_pass_1_2_PUGAN/output_xyz/pred_ply_data_test0.h5"
    # data_h5_dataset_name = "data"
    #
    # add_to_h5 = "checkpoint/dense_channel_split_high_pass_1_2_PUGAN/output_xyz/ply_data_test0.h5"
    # add_to_h5_dataset_name = "pred"
    #
    # h5_add_h5_file(data_h5=data_h5, data_h5_dataset_name=data_h5_dataset_name,
    #                add_to_h5=add_to_h5, add_to_h5_dataset_name=add_to_h5_dataset_name)


def delete_dataset_from_hdf5(file_path, dataset_name):
    """
    从HDF5文件中删除指定名称的数据集。

    参数:
    - file_path (str): HDF5文件的路径。
    - dataset_name (str): 要删除的数据集的名称。

    返回:
    - deleted (bool): 如果成功删除数据集，则为True；否则为False。
    """
    try:
        # 打开HDF5文件以进行读写操作
        with h5py.File(file_path, 'a') as hdf_file:
            # 检查数据集是否存在
            if dataset_name in hdf_file:
                # 使用del删除数据集
                del hdf_file[dataset_name]
                return True
            else:
                print(f"Dataset '{dataset_name}' not found in the file.")
                return False
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

# # 示例用法    注意打开的时候无法删除
# from utils.my_utils import delete_dataset_from_hdf5
# file_path = 'checkpoint/dense_channel_split_high_pass_1_2_PUGAN/output_xyz/ply_data_test1.h5'
# dataset_name = "pred"
# deleted = delete_dataset_from_hdf5(file_path, dataset_name)
# if deleted:
#     print(f"Dataset '{dataset_name}' deleted successfully.")
# else:
#     print(f"Failed to delete dataset '{dataset_name}'.")








# 示例用法
if __name__ == "__main__":
    h5_file_path = "../dataset/train/PC2-PU.h5"  # 替换成你的.h5文件路径
    loaded_data = read_h5_file(h5_file_path)
    if loaded_data is not None:
        print("成功读取.h5文件中的数据。")
        # 现在，loaded_data包含了从.h5文件中读取的数据，你可以根据需要使用它。
    else:
        print("无法读取.h5文件中的数据。")
