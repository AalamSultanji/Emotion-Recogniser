## Author: Aalam Sultanji
## Last updated: 25-04-2026
## Email: aalam.sultanji@gmail.com

'''
Arhictecture of the Convolutional Neural Network, contains the convolutional layers, pooling layers and fully connected layers.
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN1, self).__init__()
        '''
        Architecture: Attempt 1:
        - 4 conv layers, 2 layers of pooling, 2 dense layers. 
        - Conv + BatchNorm + ReLU activation, soft max output
        - Dropout = 0.5
        - Max Pooling = 2x2
        - GAP instead of flatten before dense layers 
        '''
        #block 1: conv + batch norm
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 32,
            kernel_size = 3,
            stride = 1, 
            padding = 1, 
            bias = False
        )
        self.bn1 = nn.BatchNorm2d(32)

        #conv block 2: conv + batch norm
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            stride = 1, 
            padding = 1, 
            bias = False
        )
        self.bn2 = nn.BatchNorm2d(64)

        #pooling layer 1: max pool
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #conv block 3: conv + batch norm
        self.conv3 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            stride = 1, 
            padding = 1, 
            bias = False
        )
        self.bn3 = nn.BatchNorm2d(128)

        #conv block 4: conv + batch norm
        self.conv4 = nn.Conv2d(
            in_channels = 128,
            out_channels = 256,
            kernel_size = 3,
            stride = 1, 
            padding = 1, 
            bias = False
        )
        self.bn4 = nn.BatchNorm2d(256)

        #pooling layer 2: max pool
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #GAP layer before dense layers
        self.gap = nn.AdaptiveAvgPool2d(1)

        #dense layer 1
        self.fc1 = nn.Linear(in_features = 256, out_features = 512)

        #dropout layer
        self.dropout = nn.Dropout(0.5)

        #dense layer 2
        self.fc2 = nn.Linear(in_features = 512, out_features = num_classes)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1) #flattens the output of the GAP layer
        x = self.fc1(x)
        x = F.relu(x) #adds non-linearity otherwise the model is just linear transform 
        x = self.dropout(x)
        x = self.fc2(x)
        return x

