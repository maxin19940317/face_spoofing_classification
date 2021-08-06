"""
测试单张图像分类
"""

import os
from tqdm import tqdm

import cv2 as cv
import torch
import torchvision.transforms as transforms

from infer import Test

torch.cuda.empty_cache()


if __name__ == '__main__':
    network_type = 'mobilenet_v2'
    weight_path = os.path.join('weights', network_type, 'model_best.pth.tar')
    im_size = 224
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet数据集计算出的均值和标准差
    ])

    test = Test(
        network_type=network_type,
        weight_path=weight_path,
        im_size=224,
        preprocess=preprocess
    )

    test_data_root = '/media/cyg/DATA1/DataSet/Face-Anti-spoofing/RITS/images/test/'
    label_names = sorted(os.listdir(test_data_root))

    for label_name in label_names:
        image_dir = os.path.join(test_data_root, label_name)
        if not os.path.isdir(image_dir):
            print(f'{image_dir} is not a directory.')
            continue

        err_cnt = 0
        images_names = sorted(os.listdir(image_dir))
        for image_name in tqdm(images_names):
            image_path = os.path.join(test_data_root, label_name, image_name)
            label_index, confidence = test.predict_single_with_align(image_path, cam=True)

            if label_index != label_names.index(label_name) and confidence > 0.9:
                err_cnt += 1
                print(image_name, label_index, confidence)

        print(f'err: {err_cnt}/{len(images_names)}, rate: {err_cnt / len(images_names)}')
