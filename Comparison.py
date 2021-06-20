import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision import models
from torchvision.models.vgg import VGG
from torch.utils.data import Dataset,TensorDataset,DataLoader
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from datetime import datetime
import torch.optim as optim
import matplotlib.pyplot as plt
#标准化图片
transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize(mean = (0.5), std = (0.5) ) ])
#定义训练集Dataset
class BagDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('train_img'))

    def __getitem__(self, idx):
        img_name = os.listdir('train_img')[idx]
        imgA = cv2.imread('train_img/' + img_name,0)
        imgB = cv2.imread('train_label/' + img_name, 0)
        if self.transform:
            imgB = self.transform(imgB)
            imgA = self.transform(imgA)
        return imgA, imgB

#定义测试集Dataset
class BagDataset1(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
    def __len__(self):
        return len(os.listdir('test_img'))
    def __getitem__(self, idx):
        img_name = os.listdir('test_img')[idx]
        imgA = cv2.imread('test_img/' + img_name,0)
        imgB = cv2.imread('test_label/' + img_name, 0)
        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return imgA, imgB
        
#生成Dataloder
bag = BagDataset(transform)
bag1=BagDataset1(transform)
train_dataset, test_dataset = bag,bag1
train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)



#定义训练参数————————————————————————————————————————————————————————————————————————————————


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#定义模型及其参数
aux_params=dict(
pooling='avg',             # one of 'avg', 'max'
dropout=0.1,               # dropout ratio, default is None
activation=None,      # activation function, default is None
classes=1,                 # define number of output labels
)
#这里可以定义PSPNet，Unet，DeepLabV3，Linknet，FPN不同的模型
#均使用了encoder_depth=5的resnet34作为encoder作为模型输入
model = smp.PSPNet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,
    classes=1,                        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    activation='sigmoid',
    encoder_depth=5
)

#定义loss，metrics以及optimizer
loss = smp.utils.losses.MSELoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5)]
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.001),])

#使用smp库封装的训练与测试器进行定义
train_epoch = smp.utils.train.TrainEpoch(
model, 
loss=loss, 
metrics=metrics, 
optimizer=optimizer,
device=DEVICE,
verbose=True)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True)

#开始训练————————————————————————————————————————————————————————————————————————————————
max_score = 0
epoch=40
for i in range(0, epoch):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_dataloader)
    valid_logs = valid_epoch.run(test_dataloader)
    #取最佳指标模型保存
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model_PSPNet.pth')
        print('Model saved!')
    #在20轮之后减小学习率以达到更加效果
    if i == 20:
        optimizer.param_groups[0]['lr'] = 1e-4
        print('Decrease decoder learning rate to 1e-4!')

