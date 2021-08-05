"""
根据视频生成训练集和测试集
1.按视频邻近帧相似度得分选取差异大的图像作为数据集
2.根据数据集类型和类别自动保存到相应目录下
"""

import os
from tqdm import tqdm
import multiprocessing

import numpy as np
import cv2 as cv

from utils.ssim_image import get_similarity_by_ssim


def preprocess_image(img, rot_flag=True):
    """获取ROI，以满足目标输入尺寸
    Args:
        img: 原始图像 -> ndarray
    Return:
        裁剪后的ROI -> ndarray(640, 480, 3)
    """
    img = img.copy()
    src_height, src_width = img.shape[:2]
    dst_height, dst_width = (640, 480)
    if src_height != dst_height:
        scale = dst_width / dst_height
        pad_height = int((src_height * scale - src_width) / (2 * scale))
        img = cv.resize(img[pad_height:-pad_height:], (dst_width, dst_height), cv.INTER_LINEAR)
    if rot_flag:
        img = np.rot90(img)
        img = np.rot90(img)
    return img


def split_video_to_image(video_path, data_type, label_name):
    """根据图像相似度，将一个视频拆分成图像
    Args:
        video_path: video所在的路径 -> str
        data_type: 数据集类型 -> str
        label_name: 数据类别名 -> str
    """
    print(f'{video_path} processing...')
    video_name = os.path.split(video_path)[-1]
    save_dir = os.path.join(image_root, data_type, label_name)
    os.makedirs(save_dir, exist_ok=True)

    rot_flag = True if label_name == 'real' else False
    # 类别print的数据集制作时移动变化较多，故最大相似度要降低以确保分类数据集均衡
    simi_score = 0.9 if label_name == 'real' else 0.8

    cap = cv.VideoCapture(video_path)
    frame_id = 1
    _, frame_a = cap.read()
    frame_a = preprocess_image(frame_a, rot_flag)
    while True:
        ret, frame_b = cap.read()
        if frame_b is None:
            break
        frame_id += 1
        frame_b = preprocess_image(frame_b, rot_flag)

        score, _ = get_similarity_by_ssim(frame_a, frame_b)

        if score < simi_score:
            frame_a = frame_b
            cv.imwrite(f'{save_dir}/{video_name[:-4]}_{frame_id}.jpg', frame_b)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    video_root = '/media/cyg/DATA1/DataSet/Face-Anti-spoofing/RITS/videos/'
    image_root = '/media/cyg/DATA1/DataSet/Face-Anti-spoofing/RITS/images_02/'
    data_types = ['train', 'test']

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count)

    for data_type in data_types:
        data_dir = os.path.join(video_root, data_type)
        label_names = sorted(os.listdir(data_dir))
        for label_name in label_names:
            video_dir = os.path.join(video_root, data_type, label_name)
            video_names = sorted(os.listdir(video_dir))
            for video_name in tqdm(video_names):
                video_path = os.path.join(video_dir, video_name)

                pool.apply_async(split_video_to_image, (video_path, data_type, label_name, ))

    pool.close()
    pool.join()
