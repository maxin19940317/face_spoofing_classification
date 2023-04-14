"""
生成训练前预处理文件
"""

import os
import yaml
import pandas as pd


class PreTrainData(object):
    def __init__(self, data_dir, label_list):
        self.data_dir = data_dir
        self.label_list = label_list

    '''
    生成csv文件
    格式为: image_name, label_index
    '''
    def gen_csv_data(self, csv_path):
        image_path_list, label_list = [], []
        for label_name in self.label_list:
            image_name_list = os.listdir(os.path.join(self.data_dir, label_name))
            for image_name in image_name_list:
                if image_name.endswith('.jpg'):
                    image_path_list.append(os.path.join(self.data_dir, label_name, image_name))
                    label_list.append(label_name)

        label_dict = self.get_label_dict()
        label_file = pd.DataFrame({'image_path': image_path_list, 'label_index': label_list})
        label_file['label_index'] = label_file['label_index'].map(label_dict)
        label_file.to_csv(csv_path, index=False)

    # 生成dict表示分类名与索引
    def get_label_dict(self):
        label_dict = {}
        for index, label in enumerate(self.label_list):
            label_dict[label] = index
        return label_dict


if __name__ == '__main__':
    os.makedirs('./data', exist_ok=True)

    with open('./configs/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_data_dir = config['DATASETS']['TRAIN_DATA_DIR']
    test_data_dir  = config['DATASETS']['TEST_DATA_DIR']
    train_label_names = sorted(os.listdir(train_data_dir))
    test_label_names  = sorted(os.listdir(test_data_dir))

    # 训练集与测试集分类须相等
    if train_label_names != test_label_names:
        assert 'please make sure train labels and test lalels are same!'

    # 生成训练集csv
    train_ptd = PreTrainData(data_dir=train_data_dir, label_list=train_label_names)
    train_ptd.gen_csv_data(csv_path=config['DATASETS']['TRAIN_CSV_FILE'])

    # 生成测试集csv
    test_ptd = PreTrainData(data_dir=test_data_dir, label_list=test_label_names)
    test_ptd.gen_csv_data(csv_path=config['DATASETS']['TEST_CSV_FILE'])
