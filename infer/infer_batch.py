import os
from tqdm import tqdm
import logging

import numpy as np
import cv2 as cv

from infer import Test

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logging.basicConfig(
    filename='./infer_print_experience.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(filename)s[line:%(lineno)3d] - %(levelname)s: %(message)s'
)


if __name__ == '__main__':
    weight_file_path = './weights/mobilenetv2_spoofing_v5.h5'
    im_size = 224
    cam = True

    test = Test(
        weight_file_path=weight_file_path,
        im_size=im_size,
    )

    test_data_root = '/media/cyg/DATA1/DataSet/Face-Anti-spoofing/RITS-640/'
    cls_names = sorted(os.listdir(test_data_root))
    for index, cls_name in enumerate(cls_names):
        print(f'cls name: {cls_name}')
        total_cnt = 0
        error_cnt = 0
        output_dir = os.path.join('outputs', cls_name)
        os.makedirs(output_dir, exist_ok=True)

        image_dir = os.path.join(test_data_root, cls_name)
        image_names = sorted(os.listdir(image_dir))
        for image_name in tqdm(image_names):
            image_path = os.path.join(image_dir, image_name)
            logging.info('-------------------------------------------')
            logging.info('image path: %s', image_path)

            # res = test.predict_single_with_one_margin(image_path, cam=True)
            res = test.predict_single_with_align(image_path, cam=True)
            if cam:
                if res:
                    img_dst = np.hstack((res['img_src'], res['img_heat_map'], res['img_fusion']))
                    image_dst_path = os.path.join(output_dir, f"{image_name[:-4]}_{res['result'][0]['name']}_{res['result'][0]['conf']}.png")
                    cv.imwrite(image_dst_path, img_dst)
                    logging.info('save cam image to %s', image_dst_path)

            if not res:
                continue
            total_cnt += 1
            if res['result'][0]['name'] != cls_name:
                logging.info('infer result: %s', res['result'])
                logging.info('pred_label: %s', res['result'][0]['name'])
                error_cnt += 1
        if total_cnt == 0:
            print('error cnt: {}, error rate: {:.3f}'.format(error_cnt, 0))
        else:
            print('error cnt: {}, error rate: {:.3f}'.format(error_cnt, error_cnt / total_cnt))
