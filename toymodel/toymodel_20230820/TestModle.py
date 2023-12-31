'''
Name:   toymodel-TestModle
Description:    This is a toymodel based on ResNet50 to apply on MNIST dataset.
Author: Yangsy
Reference:  https://blog.csdn.net/weixin_42888638/article/details/122021648
Date of creation:   2023-8-20 15:30
Last modified:  2023-8-27
'''


import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image




def init_env():
    #判断是否有GPU环境
    print(f'是否是GPU版本： {torch.cuda.is_available()}')
    print(f'使用的显卡为： {torch.cuda.get_device_name(0)}')

    #设置代码运行设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 64
    train_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='data',
                                  train=False,
                                  transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    print(len(test_dataset))
    print(test_dataset[0])
    print(test_dataset[0][0].shape)

    # Checking the dataset
    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

    return DEVICE, train_loader, test_loader




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        # x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


def resnet50(num_classes):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=Bottleneck,
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=True)
    return model


def compute_accuracy_and_loss(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        cross_entropy += F.cross_entropy(logits, targets).item()
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100, cross_entropy / num_examples


def main():

    #判断是否可以进行训练
    DEVICE, train_loader, test_loader = init_env()

    # Model class must be defined somewhere
    model = torch.load('./toymodle')
    model.eval()

    # #加载模型
    # NUM_EPOCHS = 6
    # model = resnet50(num_classes=10)
    # model = model.to(DEVICE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #
    # valid_loader = test_loader
    #
    # start_time = time.time()
    # train_acc_lst, valid_acc_lst = [], []
    # train_loss_lst, valid_loss_lst = [], []
    #
    # #开启训练
    # for epoch in range(NUM_EPOCHS):
    #     model.train()
    #     for batch_idx, (features, targets) in enumerate(train_loader):
    #
    #         ### PREPARE MINIBATCH
    #         features = features.to(DEVICE)
    #         targets = targets.to(DEVICE)
    #
    #         ### FORWARD AND BACK PROP
    #         logits, probas = model(features)
    #         cost = F.cross_entropy(logits, targets)
    #         optimizer.zero_grad()
    #
    #         cost.backward()
    #
    #         ### UPDATE MODEL PARAMETERS
    #         optimizer.step()
    #
    #         ### LOGGING
    #         if not batch_idx % 200:
    #             print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | '
    #                   f'Batch {batch_idx:04d}/{len(train_loader):04d} |'
    #                   f' Cost: {cost:.4f}')
    #
    #     # no need to build the computation graph for backprop when computing accuracy
    #     model.eval()
    #     with torch.set_grad_enabled(False):
    #         train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, device=DEVICE)
    #         valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, device=DEVICE)
    #         train_acc_lst.append(train_acc)
    #         valid_acc_lst.append(valid_acc)
    #         train_loss_lst.append(train_loss)
    #         valid_loss_lst.append(valid_loss)
    #         print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} Train Acc.: {train_acc:.2f}%'
    #               f' | Validation Acc.: {valid_acc:.2f}%')
    #
    #     elapsed = (time.time() - start_time) / 60
    #     print(f'Time elapsed: {elapsed:.2f} min')
    #
    # elapsed = (time.time() - start_time) / 60
    # print(f'Total Training Time: {elapsed:.2f} min')

    #测试
    # model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference
        test_acc, test_loss = compute_accuracy_and_loss(model, test_loader, DEVICE)
        print(f'Test accuracy: {test_acc:.2f}%')

    for features, targets in test_loader:
        break

    # 预测环节
    _, predictions = model.forward(features[:8].to(DEVICE))
    predictions = torch.argmax(predictions, dim=1)
    print(predictions)

    features = features[:7]
    fig = plt.figure()
    # print(features[i].size())
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        tmp = features[i]
        plt.imshow(np.transpose(tmp, (1, 2, 0)))
        plt.title("Actual value: {}".format(targets[i]) + '\n' + "Prediction value: {}".format(predictions[i]), size=10)

    #     plt.title("Prediction value: {}".format(tname[targets[i]]))
    plt.show()





if __name__=='__main__':
    main()

