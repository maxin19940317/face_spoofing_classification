"""
使用tf.keras训练多分类器
在Imagenet预训练权重上做迁移学习
"""

import os

import tensorflow as tf
# from tensorflow.keras.applications.resnet import ResNet50, ResNet101, ResNet152, preprocess_input
# from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2, preprocess_input
# from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# from tensorflow.keras.applications.mobilenet_v3 import MobileNetV3Large, preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from parse_args import parse_args


class Train(object):
    def __init__(self, config):
        self.train_data_root = config.train_data_root
        self.valid_data_root = config.valid_data_root
        self.network_type = config.network_type
        self.epochs = config.epochs
        self.im_size = config.image_size
        self.batch_size = config.batch_size
        self.seed = config.seed
        self.weight_dir = config.weight_dir

        self.train_generator, self.valid_generator = self.load_data()
        self.class_dict = self.train_generator.class_indices

        self.model = self.build_model()
        print(self.model.summary())

    def build_model(self):
        if self.network_type == 'resnet50_v1':
            base_model = ResNet50(
                include_top=False,  # 网络结构的最后一层,resnet50有1000类,去掉最后一层
                weights='imagenet',  # 加载预训练权重。
                input_shape=(self.im_size, self.im_size, 3),
                pooling='max',
            )
        elif self.network_type == 'resnet50_v2':
            base_model = ResNet50V2(
                include_top=False,  # 网络结构的最后一层,resnet50有1000类,去掉最后一层
                weights='imagenet',  # 加载预训练权重。
                input_shape=(self.im_size, self.im_size, 3),
                pooling='max',
            )
        elif self.network_type == 'resnet101_v2':
            # 定义ResNet101中101层初始化参数均不变
            base_model = ResNet101V2(
                include_top=False,  # 网络结构的最后一层,resnet50有1000类,去掉最后一层
                weights='imagenet',  # 加载预训练权重。
                input_shape=(self.im_size, self.im_size, 3),
                pooling='max',
            )
        elif self.network_type == 'mobilenet':
            base_model = MobileNet(
                include_top=False,  # 网络结构的最后一层,resnet50有1000类,去掉最后一层
                weights='imagenet',  # 加载预训练权重。
                input_shape=(self.im_size, self.im_size, 3),
                pooling='max',
            )
        elif self.network_type == 'mobilenet_v2':
            base_model = MobileNetV2(
                input_shape=(self.im_size, self.im_size, 3),
                include_top=False,
                weights='imagenet',
            )
        elif self.network_type == 'mobilenet_v3_large':
            base_model = MobileNetV3Large(
                input_shape=(self.im_size, self.im_size, 3),
                include_top=False,  # 网络结构的最后一层,resnet50有1000类,去掉最后一层
                weights='imagenet',  # 加载预训练权重。
            )
        elif self.network_type == 'mobilenet_v3_small':
            base_model = MobileNetV3Small(
                input_shape=(self.im_size, self.im_size, 3),
                include_top=False,  # 网络结构的最后一层,resnet50有1000类,去掉最后一层
                weights='imagenet',  # 加载预训练权重。
            )
        else:
            raise Exception('unsupported model type')

        # 冻结权重参数
        for layer in base_model.layers:
            print(layer.name)
            layer.trainable = False

        # 定义新模型
        model = base_model.layers[-3].output
        model = layers.GlobalAveragePooling2D()(model)
        model = layers.Dense(len(self.class_dict), activation=tf.nn.softmax)(model)
        model = tf.keras.Model(base_model.input, model)
        model.summary()

        model.compile(
            loss="categorical_crossentropy",
            optimizer='sgd',
            metrics=['accuracy']
        )

        return model

    def load_data(self):
        train_data_aug = ImageDataGenerator(
            preprocessing_function=preprocess_input,  # 数据做归一化
            rotation_range=5,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        )
        valid_data_aug = ImageDataGenerator(
            preprocessing_function=preprocess_input,
        )

        train_generator = train_data_aug.flow_from_directory(
            self.train_data_root,
            target_size=(self.im_size, self.im_size),
            batch_size=self.batch_size,
            seed=self.seed,
            class_mode="categorical"  # 控制目标值label的形式-选择onehot编码后的形式
        )
        valid_generator = valid_data_aug.flow_from_directory(
            self.valid_data_root,
            target_size=(self.im_size, self.im_size),
            batch_size=self.batch_size,
            shuffle=False,
            class_mode="categorical",
        )
        return train_generator, valid_generator

    def train(self):
        self.model.fit_generator(
            generator=self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            epochs=self.epochs,
            validation_data=self.valid_generator,
            validation_steps=self.valid_generator.samples // self.batch_size,
            # verbose=0,    # 日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行
            callbacks=[
                TensorBoard(
                    log_dir='./',
                    write_graph=False
                ),
                ReduceLROnPlateau(
                    monitor='loss',
                    factor=0.5,  # 学习率下降系数
                    patience=5,
                    verbose=1,
                ),
                ModelCheckpoint(
                    filepath=os.path.join(self.weight_dir + 'mobilenetv2_spoofing.{epoch:02d}-{loss:.2f}-{val_loss:.5f}.h5'),
                    monitor='loss',
                    save_best_only=True,
                    save_weights_only=False,
                ),
            ],
        )


if __name__ == '__main__':
    args = parse_args()

    trainer = Train(args)
    trainer.train()
