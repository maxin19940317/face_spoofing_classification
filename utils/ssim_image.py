from skimage.metrics import structural_similarity
import numpy as np
import cv2 as cv


def get_similarity_by_ssim(img1_rgb, img2_rgb):
    """对比图像相似度
    Args:
        img1_rgb: 原始图像1 -> ndarray
        img2_rgb: 原始图像2 -> ndarray
    Return:
        两张图像的相似度得分 -> float
        差分图 -> ndarray
    """
    assert img1_rgb.shape == img2_rgb.shape

    img1_gray = cv.cvtColor(img1_rgb, cv.COLOR_RGB2GRAY)
    img2_gray = cv.cvtColor(img2_rgb, cv.COLOR_RGB2GRAY)
    (score, diff) = structural_similarity(img1_gray, img2_gray, full=True)

    img_diff = (diff * 255).astype(np.uint8)
    return score, img_diff
