import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import supervised_convnet
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
from sklearn.model_selection import train_test_split

w1_9x9_to_3x3 = (torch.load("9x9->3x3.pt"))

class RenormalizerConvNet(nn.Module):
    def __init__(self, filter_size, square_size):
        super(RenormalizerConvNet, self).__init__()
        self.filter_size = filter_size
        self.square_size = square_size
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
        self.conv2d = nn.Conv2d(1, 1, filter_size, padding=0, stride = filter_size)  
        self.conv2d.weight = torch.nn.Parameter(w1_9x9_to_3x3['conv2d.weight'])
        self.conv2d.bias = torch.nn.Parameter(w1_9x9_to_3x3['conv2d.bias'])
        self.linear1 = nn.Linear(filter_size ** 2, 1)
        # self.linear2 = nn.Linear(100, 1)
        

    def forward(self, x):
        # add hidden layers with relu activation function
        layer1 = torch.tanh(self.conv2d(x))
        reshape = layer1.view(-1, 1, self.square_size)
        layer2 = torch.tanh(self.linear1(reshape))
        return reshape, layer2

model = RenormalizerConvNet(3, 27)

for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)

