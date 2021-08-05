import os
import time
import logging
from copy import deepcopy

import numpy as np
import cv2 as cv
from utils.align_eyes import align_eyes

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

logging.basicConfig(
    filename='./infer_experience.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(filename)s[line:%(lineno)3d] - %(levelname)s: %(message)s'
)


class Test(object):
    def __init__(self, weight_file_path, im_size=224):
        self.weight_file_path = weight_file_path
        self.label_dicts = {'print': 0, 'real': 1}
        self.im_size = im_size
        self.device = 'cuda'
        self.face_detector = self.load_face_detector(detect_face_type='retinaface')
        self.model = self.load_model()

    def load_face_detector(self, detect_face_type='centerface', device='cuda'):
        face_detector = None
        if detect_face_type == 'retinaface':
            from retinaface.retinaface import RetinaFace
            face_detector = RetinaFace(num_threads=4, device=self.device)
        elif detect_face_type == 'centerface':
            from centerface.centerface import CenterFace
            face_detector = CenterFace()
        logging.info('load face detector on %s successfully.', device)
        return face_detector

    def load_model(self):
        model = tf.keras.models.load_model(self.weight_file_path)
        # model.summary()
        logging.info('load spoofing model successfully.')
        return model

    def detect_max_face(self, img):
        start = time.time()
        face_objects = self.face_detector(img)
        logging.info('detect face time: %.3f', time.time() - start)

        if not face_objects:
            return None

        max_face = max(face_objects, key=lambda face_obj: face_obj.rect.w)

        return max_face

    def predict_single_with_one_margin(self, image_path, cam=True):
        if not os.path.isfile(image_path):
            logging.error('%s does not exist.')

        img_src = cv.imread(image_path)
        logging.info('load %s successful.', image_path)
        max_face = self.detect_max_face(img_src)
        if not max_face or max_face.prob < 0.9:
            logging.info('There are not any faces.')
            return None

        face_conf, x, y, w, h = max_face.prob, max_face.rect.x, max_face.rect.y, max_face.rect.w, max_face.rect.h
        logging.info('max face info: [%f, %f, %f, %f, %f]', face_conf, x, y, w, h)

        rate = 0.2
        margin_w, margin_h = w * rate, h * rate
        x_tl = max(int(x - margin_w / 2), 0)
        y_tl = max(int(y - margin_h / 2), 0)
        x_br = min(int(x + w + margin_w / 2), img_src.shape[1])
        y_br = min(int(y + h + margin_h / 2), img_src.shape[0])

        img_roi = img_src[y_tl:y_br, x_tl:x_br]
        img_array = cv.resize(img_roi, (224, 224), interpolation=cv.INTER_LINEAR)

        img_copy = deepcopy(img_array[:, :, ::-1])
        img_copy = np.array(img_copy).astype(np.float32)
        img_input = preprocess_input(tf.expand_dims(img_copy, 0))

        start = time.time()
        preds = self.model.predict(img_input)
        logging.info('classification time: %3f', time.time() - start)
        logging.info('res: [rate: %.2f, result: %s]', rate, preds[0])

        max_score = max(preds[0])
        if max_score < 0.9:
            return None

        result = [{'name': label_name, 'conf': round(score, 2)} for label_name, score in
                  zip(self.label_dicts.keys(), preds[0])]

        # 预测结果按置信度从大到小排列
        result = sorted(result, key=lambda item: item['conf'], reverse=True)

        if not cam:
            return {'result': result}
        else:
            gt_label = image_path.split('_')[1]
            img_heat_map = self.calc_cam(img_input, gt_label)

            # superimposing heatmap and image
            img_fusion = img_heat_map * 0.5 + img_copy
            img_fusion = np.clip(img_fusion, 0, 255).astype(np.uint8)

            return {'result': result, 'img_src': img_array, 'img_heat_map': img_heat_map, 'img_fusion': img_fusion}

    def predict_single_with_align(self, image_path, cam=True):
        if not os.path.isfile(image_path):
            logging.error('%s does not exist.')

        img_src = cv.imread(image_path)
        logging.info('load %s successful.', image_path)
        max_face = self.detect_max_face(img_src)
        if not max_face or max_face.prob < 0.9:
            logging.info('There are not any faces.')
            return None
        # print(max_face.prob)

        left_eye_point = (max_face.landmark[0].x, max_face.landmark[0].y)
        right_eye_point = (max_face.landmark[1].x, max_face.landmark[1].y)

        img_aligned = align_eyes(img_src, left_eye_point, right_eye_point)
        img_array = cv.resize(img_aligned, (224, 224), interpolation=cv.INTER_LINEAR)

        img_copy = deepcopy(img_array[:, :, ::-1])
        img_copy = np.array(img_copy).astype(np.float32)
        img_input = preprocess_input(tf.expand_dims(img_copy, 0))

        start = time.time()
        preds = self.model.predict(img_input)
        logging.info('classification time: %3f', time.time() - start)
        logging.info('res: [result: %s]', preds[0])

        max_score = max(preds[0])
        if max_score < 0.9:
            return None

        result = [{'name': label_name, 'conf': round(score, 2)} for label_name, score in
                  zip(self.label_dicts.keys(), preds[0])]

        # 预测结果按置信度从大到小排列
        result = sorted(result, key=lambda item: item['conf'], reverse=True)

        if not cam:
            return {'result': result}
        else:
            gt_label = image_path.split('_')[1]
            img_heat_map = self.calc_cam(img_input, gt_label)

            # superimposing heatmap and image
            img_fusion = img_heat_map * 0.5 + img_copy
            img_fusion = np.clip(img_fusion, 0, 255).astype(np.uint8)

            return {'result': result, 'img_src': img_array, 'img_heat_map': img_heat_map, 'img_fusion': img_fusion}

    def calc_cam(self, img_input, gt_label):
        # getting class weights for last layer in model
        class_weights = self.model.layers[-1].get_weights()[0]

        # commening process of getting class weights for last convolutional layer by first defining it
        final_conv_layer = self.model.layers[-3]

        # defining a backend function to get outputs for various layers in the model
        get_output = tf.keras.backend.function([self.model.layers[0].input], [final_conv_layer.output])

        # getting outputs for each target file
        [conv_outputs] = get_output(img_input)
        conv_outputs = conv_outputs[0, :, :, :]

        # initializing a matrix to store class activation map
        cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])

        # iterating through weights and adding them to activation map
        gt_label_index = self.label_dicts[gt_label]
        for index, weight in enumerate(class_weights[:, gt_label_index]):
            cam += weight * conv_outputs[:, :, index]

        # normalizing activation map
        cam = np.maximum(cam, 0)
        cam /= np.max(cam)

        # postprocessing heatmap
        cam = cv.resize(cam, (224, 224))
        img_heatmap = cam * 255
        img_heatmap = np.clip(img_heatmap, 0, 255).astype(np.uint8)
        img_heatmap = cv.applyColorMap(img_heatmap, cv.COLORMAP_JET)

        return img_heatmap


if __name__ == '__main__':
    weight_file_path = './weights/mobilenetv2_spoofing.31-0.00-0.13022.h5'
    im_size = 224
    cam = True

    test = Test(
        weight_file_path=weight_file_path,
        im_size=im_size,
    )

    test_image_path = '/media/cyg/DATA1/DataSet/Face-Anti-spoofing/RITS/images/test/real/cyg_real_mask_glasses_brightness_30.jpg'
    logging.info('-------------------------------------------')
    logging.info('image path: %s', test_image_path)
    res = test.predict_single_with_one_margin(test_image_path, cam=True)
    if cam:
        if res:
            img_dst = np.hstack((res['img_src'], res['img_heat_map'], res['img_fusion']))
            logging.info('save dst.png to %s', 'dst.png')
            cv.imwrite('dst.png', img_dst)

    logging.info('infer result: %s', res)
    logging.info('-------------------------------------------')
