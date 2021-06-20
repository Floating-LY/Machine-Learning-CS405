#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os 
import time
import tqdm
import torch
import random
import numpy as np
import pandas as pd 
import torch.nn as nn
from PIL import Image
import matplotlib
import matplotlib.image as img
from torchvision import datasets, models, transforms

# ---Initialization of super parameters and model settings---
def get_args():
    rootpath = ""                                       #rootpath of data
    args = {
        'batch_size': 32,
        'Epochs':300,
        'Lr': 0.0003,#0.003,
        'trainfold': rootpath + 'dataset/train_img/',
        'trainlabel': rootpath + 'dataset/train_label/',
        'testfold': rootpath + 'dataset/test_img/',
        'testlabel': rootpath + 'dataset/test_label/',
        'mark': 'vertical_lr0003_channelplus',
        'flip': "v" # flip:  "":no flip, "v":vertical, "h":horizron              
    }
    return args

# ---Read in data---
class Dataset(torch.utils.data.Dataset):
    def __init__(self, datafold, label, idx, is_train=True, flip=""):
        self.datafold = datafold
        self.label = label
        self.idx = idx
        self.is_train = is_train
        self.totensor = transforms.ToTensor()
        self.flip = flip
        

    def __getitem__(self, i):

        index = self.idx[i%len(self.idx)]
        flip = 0 # no flip
        if i+1 >= len(self.idx): #If flipped, the dataset be doubled, latter half be flipped
            flip = 1 if self.flip == "v" else 2            
        
        img = torch.flip(self.totensor(Image.open(self.datafold + str(index) + ".png")), [flip])
      
        
        if self.is_train:
            label = np.load(self.label+ str(index) + ".png.npy")
            #label = torch.as_tensor(label, dtype=torch.float32)
            label2 = self.totensor(Image.open(self.label+ str(index) + ".png"))
            label2 = torch.flip(label2, [flip])
                         
            return img, label2
        else: 
            label2 = self.totensor(Image.open(self.label+ str(index) + ".png").convert('L'))    
            return img, label2

    def __len__(self):
        if self.flip != "":
            return len(self.idx) * 2 #flip, doubled
        else:
            return len(self.idx)
 
 #---Defining the basic block of CNN---       
class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,dilation=dilation,stride=stride,padding=padding, bias=False),
            nn.ReLU()
            )
        self.right = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1, bias=False),
            nn.ReLU()
            )
        self.out=nn.ReLU()
    def forward(self, x):
        out = self.left(x)
        residual = self.right(x)
        out += residual
        return self.out(out)

class CNN(nn.Module):
    def __init__(self):
        super(simple,self).__init__()
        self.Conv_ReLU_1=ResidualBlock(in_channels=1,out_channels=128,kernel_size=5,stride=1,padding=2)
        
        # layer with dilated kernel
        self.Conv_ReLU_2=ResidualBlock(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=5,dilation=5)
        
        self.Conv_ReLU_3=ResidualBlock(in_channels=256,out_channels=512,kernel_size=5,stride=1,padding=2)
        
        # layer with dilated kernel
        self.Conv_ReLU_4=ResidualBlock(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=5,dilation=5)
        self.Conv=nn.Conv2d(in_channels=512,out_channels=1,kernel_size=1,stride=1)
        self.out=nn.Sigmoid()  
    def forward(self,x):
        x=self.Conv_ReLU_1(x)
        x=self.Conv_ReLU_2(x)
        x=self.Conv_ReLU_3(x)
        x=self.Conv_ReLU_4(x)
        x=self.Conv(x)   # 
        x=self.out(x)
        return x


if __name__ == '__main__':
    #---Fixed seed
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
      
    setup_seed(29)

    args = get_args()
    
    #---get dataloader
    train_idx = [ i for i in range(25)]
    train_dataset = Dataset(args["trainfold"], args["trainlabel"], train_idx, is_train=True, flip=args["flip"])
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                        shuffle=True,
                                        batch_size=args["batch_size"])
    
    test_idx =[ i for i in range(5)]
    test_dataset = Dataset(args["testfold"], args["testlabel"], test_idx, is_train=False)    
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                        shuffle=False,
                                        batch_size=args["batch_size"])

    #---get model, loss function, optimizer
    model = CNN()

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)

        
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print('divice:', device)
    model = model.to(device)
    
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args['Lr'])
     
    #---define train_test function
    def Train_Test(model, loss_function, optimizer, train_dataloader, test_dataloader, epochs):

        history = []
        best_loss = 1000.0
        best_acc = 0.0
        best_epoch = 0
      
        test_acc = 0.0
            
        for epoch in range(epochs):
            epoch_start = time.time()
  
            print("Epoch: {}/{} train".format(epoch+1, epochs))
            model.train()
           
            for images, labels2 in train_dataloader:
                images = images.to(device)
                labels = labels2.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss = loss.item()
      
                pred = (outputs >= 0.5) + 0   
                correct_counts = pred.eq(labels)
                train_acc = torch.mean(correct_counts.type(torch.FloatTensor)).item()

            
            with torch.no_grad():
                model.eval()
                for images, labels in test_dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)

                        
                    
                    pred = (outputs >= 0.5) + 0
                    correct_counts = pred.eq(labels)
                    test_acc += torch.mean(correct_counts.type(torch.FloatTensor)).item()
                    

            history.append(train_loss)
            #---save model based on acc 
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
                torch.save(model, './{}_best_model_1c.pth'.format(args["mark"]))

                    

            epoch_end = time.time()

            print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tTest: Accuracy: {:.4f}%,  Best Accuracy for test : {:.4f} at epoch {:03d}".format(epoch+1, train_loss, train_acc, test_acc, best_acc, best_epoch))
         
        #---predict 
        model = torch.load("{}_best_model_1c.pth".format(args["mark"]))
        with torch.no_grad():
            model.eval()
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                    
                outputs = model(images)
                outputs_np = outputs.squeeze(1).cpu().detach().numpy()
                for i in range(outputs_np.shape[0]):
                    i_np = np.fix(255 * outputs_np[i])
                    matplotlib.image.imsave("outputs/out_{}_{}.png".format(args["mark"], i), i_np, cmap="gray")
                   
        return model, history


        
    Ts = time.time()
    trained_model, history = Train_Test(model, loss_func, optimizer, train_dataloader, test_dataloader, args['Epochs'])
    Te = time.time() - Ts
