import os
import shutil
import yaml
import logging
import logging.config

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

logging.config.fileConfig("./configs/logging.conf")
logging = logging.getLogger('root')


class TrainDataset(Dataset):
    def __init__(self, item_list, transform=None):
        images = []
        for index, row in item_list.iterrows():
            images.append((row['image_path'], row['label_index']))
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        image_path, label = self.images[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)


# 用于计算精度和时间的变化
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 计算top K准确率
def accuracy(y_pred, y_actual, topk=(1,)):
    pred_count = y_actual.size(0)
    pred_correct_count = 0
    prob, pred = y_pred.topk(max(topk), 1, True, True)

    for j in range(pred.size(0)):
        if int(y_actual[j]) == int(pred[j]):
            pred_correct_count += 1
    if pred_count == 0:
        final_acc = 0
    else:
        final_acc = pred_correct_count / pred_count
    return final_acc * 100, pred_count


class Train(object):
    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_type = config['NETWORK']['NETWORK_TYPE']
        self.label_num = config['DATASETS']['LABEL_NUM']
        self.lr = config['TRAIN']['LR']
        self.lr_decay = config['TRAIN']['LR_DECAY']
        self.weight_decay = config['TRAIN']['WEIGHT_DECAY']
        self.stage_epochs = config['TRAIN']['STAGE_EPOCHS']
        self.epochs = config['TRAIN']['EPOCHS']
        self.start_epoch = 0
        self.batch_size = config['TRAIN']['BATCH_SIZE']
        self.stage = 0
        self.model = self.build_model()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                    amsgrad=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.stage_epochs, gamma=0.5)

        self.best_precision = 0
        self.lowest_loss = 1e3

    def build_model(self):
        if self.model_type == 'mobilenet_v2':
            from networks.mobilenetv2 import MobileNetV2
            model = MobileNetV2(num_classes=self.label_num)
            model = model.to(self.device)
        else:
            model = None
        return model

    # 加载预训练模型参数
    def resume_train(self):
        checkpoint_path = f'./weights/{self.model_type}/checkpoint.pth.tar'
        if not os.path.isfile(checkpoint_path):
            print("=> no checkpoint found at {checkpoint_path}")
        else:
            print(f"=> loading checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_precision = checkpoint['best_precision']
            self.lowest_loss = checkpoint['lowest_loss']
            self.stage = checkpoint['stage']
            self.lr = checkpoint['lr']
            self.model.load_state_dict(checkpoint['state_dict'])

            # 如果中断点恰好为转换stage的点，需要特殊处理
            if self.start_epoch in np.cumsum(self.stage_epochs)[:-1]:
                self.stage += 1
                self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr / self.lr_decay,
                                            weight_decay=self.weight_decay, amsgrad=True)
                self.model.load_state_dict(
                    torch.load(f'./weights/{self.model_type}/model_best.pth.tar')['state_dict'])
            print(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")

    # 加载训练集
    def load_train_data(self):
        train_data = pd.read_csv('data/train.csv')

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_data = TrainDataset(item_list=train_data, transform=preprocess)

        # 生成图片迭代器
        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=4, drop_last=True)
        return train_loader

    def save_checkpoint(self, state, is_best, is_lowest_loss):
        """
        保存最新模型以及最优模型
        """
        model_path = f'./weights/{self.model_type}/checkpoint.pth.tar'
        torch.save(state, model_path)
        if is_best:
            shutil.copyfile(model_path, f'./weights/{self.model_type}/model_best.pth.tar')
        if is_lowest_loss:
            shutil.copyfile(model_path, f'./weights/{self.model_type}/lowest_loss.pth.tar')

    def train(self):
        if config['TRAIN']['RESUME']:
            self.resume_train()

        train_loader = self.load_train_data()
        criterion = nn.CrossEntropyLoss().cuda(0)

        # 开始训练
        for epoch in range(self.start_epoch, self.epochs):
            logging.info('lr: %.5f', float(self.scheduler.get_last_lr()[0]))

            losses = AverageMeter()
            acc = AverageMeter()

            self.model.train()

            # 从训练集迭代器中获取训练数据
            for index, (images, target) in enumerate(train_loader):
                # 将图片和标签转化为tensor
                image_var = images.clone().detach().cuda()
                label = target.clone().detach().cuda()

                # 将图片输入网络，前传，生成预测值
                y_pred = self.model(image_var)
                loss = criterion(y_pred, label)
                losses.update(loss.item(), images.size(0))

                # 计算正确率
                prec, pred_count = accuracy(y_pred.data, target, topk=(1, 1))
                acc.update(prec, pred_count)

                # 对梯度进行反向传播，使用随机梯度下降更新网络权重
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 打印耗时与结果
                logging.info('Epoch: [{0}][{1}/{2}]\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, index, len(train_loader),
                                                                            loss=losses,
                                                                            acc=acc))
            precision = acc.avg
            avg_loss = losses.avg

            # 记录最高精度与最低loss，保存最新模型与最佳模型
            is_best = precision > self.best_precision
            is_lowest_loss = avg_loss < self.lowest_loss
            self.best_precision = max(precision, self.best_precision)
            self.lowest_loss = min(avg_loss, self.lowest_loss)
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_precision': self.best_precision,
                'lowest_loss': self.lowest_loss,
                'stage': self.stage,
                'lr': self.lr,
            }
            self.scheduler.step()
            self.save_checkpoint(state, is_best, is_lowest_loss)


if __name__ == '__main__':
    with open('./configs/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    network_type = config['NETWORK']['NETWORK_TYPE']
    os.makedirs(f'./weights/{network_type}', exist_ok=True)
    os.makedirs(f'./weights/{network_type}', exist_ok=True)

    tm = Train(config)
    tm.train()
