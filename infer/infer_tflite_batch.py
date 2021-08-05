import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import time

test_data_root = './data/test_data/'
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)
cls_names = os.listdir(test_data_root)

image_info_list = []
for index, cls_name in enumerate(cls_names):
    label = cls_name
    image_dir = os.path.join(test_data_root, cls_name)
    image_names = os.listdir(image_dir)
    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        image_info_list.append({'image_path': image_path, 'label': index})
print('image total:', len(image_info_list))

interpreter = tf.lite.Interpreter(model_path='./weights/mobilenetv2_spoofing_optimize_float16.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

time_list = []
error_cnt = 0
for image_info in image_info_list:
    img = image.load_img(
        image_info['image_path'],
        target_size=(224, 224)
    )
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = preprocess_input(img_array)

    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    time_list.append(time.time() - start)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if np.argmax(output_data) != image_info['label']:
        error_cnt += 1
        print(np.argmax(output_data), np.max(output_data))
print(error_cnt / (291 + 1056))
print(np.average(time_list))
