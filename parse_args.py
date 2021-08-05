import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_root",
        help="train data root",
        default='/media/cyg/DATA1/DataSet/Face-Anti-spoofing/RITS/data_aligned_02/train_data',
        type=str
    )
    parser.add_argument(
        "--valid_data_root",
        help="valid data root",
        default='/media/cyg/DATA1/DataSet/Face-Anti-spoofing/RITS/data_aligned_02/test_data',
        type=str
    )
    parser.add_argument(
        "--network_type",
        help="network type [resnet50_v1, resnet50_v2, resnet101_v2, mobilenet, mobilenet_v2, mobilenet_v3_large]",
        default='mobilenet_v2',
        type=str
    )
    parser.add_argument(
        "--image_size",
        help="image size",
        default=224,
        type=int
    )
    parser.add_argument(
        "--epochs",
        help="epochs",
        default=100,
        type=int
    )
    parser.add_argument(
        "--batch_size",
        help="batch size",
        default=16,
        type=int
    )
    parser.add_argument(
        "--seed",
        help="fixed seed num",
        default=1024,
        type=int
    )
    parser.add_argument(
        "--weight_dir",
        help="save weight dir",
        default='./weights/',
        type=str
    )
    return parser.parse_args()
