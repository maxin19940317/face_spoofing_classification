import os
import time
import logging
from copy import deepcopy

import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary

from networks.mobilenetv2 import MobileNetV2
from utils.align_eyes import align_eyes

torch.cuda.empty_cache()


class Test(object):
    def __init__(self, network_type: str,
                 weight_path: str,
                 im_size=224,
                 preprocess=None):
        self.network_type = network_type
        self.weight_path = weight_path
        self.im_size = im_size
        self.preprocess = preprocess
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.face_detector = self.load_face_detector(detect_face_type='retinaface')
        self.model = self.load_model()
        self.feature = None
        self.gradient = None
        self.handlers = []

    def load_face_detector(self, detect_face_type='centerface'):
        """加载人脸检测器
        Args:
            detect_face_type: 人脸检测器框架(retinaface, centerface)

        Returns:人脸检测器
        """
        face_detector = None
        if detect_face_type == 'retinaface':
            from retinaface.retinaface import RetinaFace
            face_detector = RetinaFace(num_threads=4, device=self.device)
        elif detect_face_type == 'centerface':
            from centerface.centerface import CenterFace
            face_detector = CenterFace()
        logging.info('load face detector on %s successfully.' % self.device)
        return face_detector

    def detect_max_face(self, img):
        """检测图像中最大人脸
        Args:
            img: 输入图像 -> ndarray

        Returns:
            人脸目标 ->　face object
        """
        start = time.time()
        face_objects = self.face_detector(img)
        logging.info('detect face time: %.3f' % (time.time() - start))
        if not face_objects:
            return None

        max_face = max(face_objects, key=lambda face_obj: face_obj.rect.w)

        return max_face

    def load_model(self):
        """加载分类模型权重参数

        Returns:分类器
        """
        if self.network_type == 'mobilenet_v2':
            model = MobileNetV2(num_classes=2)
        else:
            model = None
            raise ValueError('%s does not match.' % self.network_type)

        summary(model, input_size=[(3, self.im_size, self.im_size)], batch_size=2, device="cpu")
        best_model = torch.load(self.weight_path)
        model.load_state_dict(best_model['state_dict'])
        model = model.to(self.device)
        return model

    def _get_last_conv_name(self):
        # 获取网络的最后一个卷积层的名字
        layer_name = None
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                layer_name = name
        return layer_name

    def _get_features_hook(self, module, input, output):
        self.feature = output

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0]

    def _register_hook(self):
        layer_name = self._get_last_conv_name()
        for (name, module) in self.model.named_modules():
            if name == layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def predict_single_with_align(self, image_path, cam=True):
        """对输入图像分类(print, real)
        Args:
            image_path: 待测试的图像　-> ndarray
            cam: 是否保存热力图 -> bool

        Returns:
            分类索引, 分类置信度
        """
        if not os.path.exists(image_path):
            logging.error('%s does not exist.' % image_path)

        img = cv.imread(image_path)
        logging.info('load %s successful.' % image_path)

        max_face = self.detect_max_face(img)
        if not max_face or max_face.prob < 0.9:
            logging.info('There are not any faces.')
            return None, 0

        left_eye_point = (max_face.landmark[0].x, max_face.landmark[0].y)
        right_eye_point = (max_face.landmark[1].x, max_face.landmark[1].y)

        img_aligned = align_eyes(img, left_eye_point, right_eye_point)
        img_aligned = cv.resize(img_aligned, (self.im_size, self.im_size), interpolation=cv.INTER_LINEAR)
        img_aligned = img_aligned[..., ::-1].copy()

        input_tensor = self.preprocess(img_aligned)
        input_batch = input_tensor.view(1, 3, self.im_size, self.im_size).to(self.device)

        self.model.eval()  # 固定BatchNormalization和Dropout
        self.model.zero_grad()
        self._register_hook()

        pred = self.model(input_batch)
        """
        将张量的每个元素缩放到(0, 1)区间且和为1
        dim为0，按列计算
        dim为1，按行计算
        """
        m = nn.Softmax(dim=1)

        prob_list = m(pred)[0].tolist()
        target_index = np.argmax(pred.cpu().data.numpy())
        target_score = max(prob_list)

        if cam:
            image_name = os.path.split(image_path)[-1]
            label_name = 'print' if target_index == 0 else 'real'
            output_dir = 'outputs'
            cam_save_path = os.path.join(output_dir, f'{image_name[:-4]}_{label_name}_{target_score}.jpg')
            self.save_cam_fusion(img_aligned[..., ::-1], pred, target_index, cam_save_path)

        return target_index, target_score

    def save_cam_fusion(self, img, pred, index, save_path):
        """计算热力图并保存
        Args:
            img: 待测试的图像　-> ndarray
            pred: 分类模型输出结果　-> tensor
            index: 分类索引 -> int
            save_path: 保存路径　-> str
        """
        target = pred[0][index]
        target.requires_grad_ = True
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()
        feature = self.feature[0].cpu().data.numpy()

        weight = np.mean(gradient, axis=(1, 2))
        img_cam_mask = feature * weight[:, np.newaxis, np.newaxis]
        img_cam_mask = np.sum(img_cam_mask, axis=0)
        img_cam_mask = np.maximum(img_cam_mask, 0)  # ReLU

        # 数值归一化
        img_cam_mask -= np.min(img_cam_mask)
        img_cam_mask /= np.max(img_cam_mask)

        img_cam_mask = cv.resize(img_cam_mask, (self.im_size, self.im_size), interpolation=cv.INTER_LINEAR)

        # postprocessing heatmap
        img_cam_mask = cv.resize(img_cam_mask, (224, 224))
        img_heatmap = img_cam_mask * 255
        img_heatmap = np.clip(img_heatmap, 0, 255).astype(np.uint8)
        img_heatmap = cv.applyColorMap(img_heatmap, cv.COLORMAP_JET)

        img_cam_fusion = img_heatmap * 0.5 + img
        img_cam_fusion = np.clip(img_cam_fusion, 0, 255).astype(np.uint8)

        img_dst = np.hstack((img, img_heatmap, img_cam_fusion))
        cv.imwrite(save_path, img_dst)


if __name__ == '__main__':
    network_type = 'mobilenet_v2'
    weight_path = os.path.join('weights', network_type, 'model_best.pth.tar')
    im_size = 224
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet数据集计算出的均值和标准差
    ])
    cam = True

    test = Test(
        network_type=network_type,
        weight_path=weight_path,
        im_size=224,
        preprocess=preprocess
    )

    test_image_path = '/media/cyg/DATA1/DataSet/Face-Anti-spoofing/RITS/data_aligned/test/real/cyg_real_mask_glasses_brightness_32.jpg'
    logging.info('-------------------------------------------')
    logging.info('image path: %s' % test_image_path)
    label_index, conf = test.predict_single_with_align(test_image_path, cam=True)
    print(label_index, conf)

    test.remove_handlers()
