import os
import argparse
import tensorflow as tf
import cv2
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--keras_model_path", help="model_path", default='./weights/mobilenetv2_spoofing.h5', type=str)
    parser.add_argument("--lite_model_path", help="outputs_layer", default=None, type=str)
    parser.add_argument("--optimize", help="optimize model", default=True, type=bool)
    parser.add_argument("--quantization", help="quantization model", default="float16", type=str)

    return parser.parse_args()


def converer_keras_to_tflite_v2(keras_path, out_tflite=None, optimize=False, quantization='int8'):
    """
    :param keras_path: keras *.h5 files
    :param outputs_layer: default last layer
    :param out_tflite: output *.tflite file
    :param optimize
    :return:
    """
    if not os.path.exists(keras_path):
        raise Exception("Error:{}".format(keras_path))
    model_dir = os.path.dirname(keras_path)
    model_name = os.path.basename(keras_path)[:-len(".h5")]
    # 加载keras模型, 结构打印
    model = tf.keras.models.load_model(keras_path)
    print(model.summary())

    converter = tf.lite.TFLiteConverter.from_keras_model(model)  # tf2.0
    prefix = [model_name]
    # converter.allow_custom_ops = False
    # converter.experimental_new_converter = True
    """"
    https://tensorflow.google.cn/lite/guide/ops_select
    我们优先推荐使用 TFLITE_BUILTINS 转换模型，然后是同时使用 TFLITE_BUILTINS,SELECT_TF_OPS ，
    最后是只使用 SELECT_TF_OPS。同时使用两个选项（也就是 TFLITE_BUILTINS,SELECT_TF_OPS）
    会用 TensorFlow Lite 内置的运算符去转换支持的运算符。
    有些 TensorFlow 运算符 TensorFlow Lite 只支持部分用法，这时可以使用 SELECT_TF_OPS 选项来避免这种局限性。
    """
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    if optimize:
        print("weight quantization")
        # Enforce full integer quantization for all ops and use int input/output
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        prefix += ["optimize"]
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantization == "int8":
        converter.representative_dataset = representative_dataset_gen
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
        converter.target_spec.supported_types = [tf.int8]
    elif quantization == "float16":
        converter.target_spec.supported_types = [tf.float16]

    prefix += [quantization]
    if not out_tflite:
        prefix = [str(n) for n in prefix if n]
        prefix = "_".join(prefix)
        out_tflite = os.path.join(model_dir, "{}.tflite".format(prefix))
    tflite_model = converter.convert()
    open(out_tflite, "wb").write(tflite_model)
    print("successfully convert to tflite done")
    print("save model at: {}".format(out_tflite))


def representative_dataset_gen():
    """
    # Generating representative data sets
    :return:
    """
    image_dir = '/media/cyg/DATA1/DataSet/ClassDataset/flower_photos_split/test_images/daisy/'
    input_size = [224, 224]
    imgSet = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    for img_path in imgSet:
        orig_image = cv2.imread(img_path)
        rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image_tensor = cv2.resize(rgb_image, dsize=tuple(input_size))
        image_tensor = np.asarray(image_tensor / 255.0, dtype=np.float32)
        image_tensor = image_tensor[np.newaxis, :]
        yield [image_tensor]


if __name__ == '__main__':
    args = parse_args()

    converer_keras_to_tflite_v2(args.keras_model_path, out_tflite=args.lite_model_path, optimize=args.optimize, quantization=args.quantization)

