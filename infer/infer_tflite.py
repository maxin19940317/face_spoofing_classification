import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import cv2
import numpy as np
import time

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

start = time.time()
test_image_path = './test.jpg'
img = image.load_img(
    test_image_path,
    target_size=(224, 224)
)
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch
img_array = preprocess_input(img_array)

interpreter = tf.lite.Interpreter(model_path='./weights/mobilenet_flowers_optimize_float16.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
for _ in range(10):
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    print(time.time()-start)
    output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

