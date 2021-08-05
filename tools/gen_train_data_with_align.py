"""
根据视频拆解的原图生成训练数据集(训练集和测试集)
1.检测最大人脸
2.人脸对齐
"""

import os
import multiprocessing

import numpy as np
import cv2 as cv

from retinaface.retinaface import RetinaFace
from utils.align_eyes import align_eyes


def detect_max_face(img):
    """检测输入的图像内最大人脸
    Args:
        img: 输入的图像 -> ndarray
    Return:
        最大人脸目标　-> face object
    """
    face_objects = net(img)
    if not face_objects:
        return None

    max_face = max(face_objects, key=lambda face_obj: face_obj.rect.w)

    return max_face


def align_face(image_path, output_dir):
    """人脸对齐
    Args:
        image_path: 输入图像的路径 -> str
    Return:
        对齐后的人脸
    """
    img_src = cv.imread(image_path)
    max_face = detect_max_face(img_src)
    if not max_face or max_face.prob < 0.9:
        return

    left_eye_point = (max_face.landmark[0].x, max_face.landmark[0].y)
    right_eye_point = (max_face.landmark[1].x, max_face.landmark[1].y)

    # 根据人眼做人脸对齐
    img_aligned = align_eyes(img_src, left_eye_point, right_eye_point)

    image_name = image_path.split('/')[-1]
    image_save_path = os.path.join(output_dir, f'{image_name[:-4]}.jpg')
    cv.imwrite(image_save_path, img_aligned)
    return


if __name__ == '__main__':
    net = RetinaFace(num_threads=1)

    input_root = '/media/cyg/DATA1/DataSet/Face-Anti-spoofing/RITS/images_02/'
    output_root = '/media/cyg/DATA1/DataSet/Face-Anti-spoofing/RITS/data_aligned_03/'
    data_types = ['train', 'test']

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count)

    for data_type in data_types:
        data_dir = os.path.join(input_root, data_type)
        label_names = sorted(os.listdir(data_dir))
        for label_name in label_names:
            image_dir = os.path.join(input_root, data_type, label_name)
            image_names = sorted(os.listdir(image_dir))

            output_dir = os.path.join(output_root, data_type, label_name)
            os.makedirs(output_dir, exist_ok=True)

            for image_name in image_names:
                if not image_name.endswith('.jpg'):
                    print(f'{image_name} is not a jpg file.')
                    continue

                image_path = os.path.join(input_root, data_type, label_name, image_name)
                pool.apply_async(align_face, (image_path, output_dir))

    pool.close()
    pool.join()
