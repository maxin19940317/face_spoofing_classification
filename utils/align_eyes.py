import numpy as np
import cv2 as cv


def align_eyes(img, left_eye_point, right_eye_point):
    """人脸对齐
    Args:
        img: 需对齐的人脸图像 -> ndarray
        left_eye_point: 左眼坐标 -> tuple
        right_eye_point: 右眼坐标 -> tuple
    Return:
        对齐后的人脸 -> ndarray
    """

    face_size = (96*2, 112*2)
    left_eye_ref = (30.29459953*2, 51.69630051*2)  # 模板左眼中心点坐标
    right_eye_ref = (65.53179932*2, 51.50139999*2)  # 模板右眼中心点坐标
    eye_dist_ref = right_eye_ref[0] - left_eye_ref[0]  # 模板左右眼中心点距离

    delta_x = right_eye_point[0] - left_eye_point[0]
    delta_y = right_eye_point[1] - left_eye_point[1]
    eye_dist = np.sqrt((delta_x ** 2) + (delta_y ** 2))  # 待校正左右眼中心距离

    scale = eye_dist_ref / eye_dist  # 各项同性缩放因子
    angle = np.degrees(np.arctan2(delta_y, delta_x))  # 旋转角度
    center = ((left_eye_point[0] + right_eye_point[0]) // 2, (left_eye_point[1] + right_eye_point[1]) // 2)  # 旋转中心

    M = cv.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += face_size[0] // 2 - center[0]
    M[1, 2] += left_eye_ref[1] - center[1]

    img_aligned = cv.warpAffine(img, M, face_size, flags=cv.INTER_LINEAR)

    return img_aligned