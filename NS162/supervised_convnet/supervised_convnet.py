import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import sys

class SupervisedConvNet(nn.Module):
    def __init__(self, filter_size, square_size):
        super(SupervisedConvNet, self).__init__()
        self.filter_size = filter_size
        self.square_size = square_size
        self.conv2d = nn.Conv2d(1, 1, filter_size, padding=0, stride = filter_size)  
        self.linear1 = nn.Linear(filter_size ** 2, 100)
        self.linear2 = nn.Linear(100, 1)
        
    def forward(self, x):
        # add hidden layers with relu activation function
        layer1 = F.relu(self.conv2d(x))
        reshape = layer1.view(-1, 1, self.square_size)
        linear = self.linear(reshape)
        layer2 = torch.sigmoid(linear)
        # for row in x:
        #     print("row", row)
        #     for el in row[0]:
        #         print("el", el)
        # x = torch.tanh(self.decoder(x))
        return conv2, layer1, reshape, linear, layer2

