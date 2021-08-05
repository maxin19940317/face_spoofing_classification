"""
用MD5去除指定文件夹下的重复图片
"""

import os
import hashlib


def get_file_list(data_dir, file_list):
    """ 递归遍历文件夹下所有图片文件
    :param data_dir: 根目录
    :param file_list: 图片名列表
    :return: file_list
    """

    if os.path.isfile(data_dir):
        if data_dir.endswith('.jpg') or data_dir.endswith('.png'):
            file_list.append(data_dir)
    elif os.path.isdir(data_dir):
        for s in os.listdir(data_dir):
            child_dir = os.path.join(data_dir, s)
            get_file_list(child_dir, file_list)
    return file_list


if __name__ == '__main__':
    data_dir = '/media/cyg/DATA1/DataSet/ClassDataset/oregon_wildlife/'
    file_list = []
    image_path_list = get_file_list(data_dir, file_list)
    img_type_list = ['.jpg', '.bmp', '.png', '.jpeg']
    hash_keys = []
    for image_path in image_path_list:
        if not os.path.isfile(image_path) or os.path.splitext(image_path)[1] not in img_type_list:
            # os.remove(image_path)
            print('not a pic %s', image_path)
            continue

        with open(image_path, 'rb') as f:
            hash_key = hashlib.md5(f.read()).hexdigest()
        if hash_key not in hash_keys:
            hash_keys.append(hash_key)
        else:
            os.remove(image_path)
            print('remove %s', image_path)
