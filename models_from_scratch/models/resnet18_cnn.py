import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels)
        )
        
        self.shortcut = nn.Identity()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.out_channels)
            )

    def forward(self, x):
        out = self.conv_block(x)
        #print(self.shortcut(x).shape, 'this is x ')
        #print(out.shape, 'this is out')
        out += self.shortcut(x)
        out = F.relu(out) 

        return out

class ResNet18(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(256, num_classes)

        self.layer1 = self._create_layer(block, num_blocks, 64, 64)
        self.layer2 = self._create_layer(block, num_blocks, 64, 128)
        self.layer3 = self._create_layer(block, num_blocks, 128, 256)
        self.layer4 = self._create_layer(block, num_blocks, 256, 256)
        
    def _create_layer(self, block, num_blocks, in_channels, out_channels):
            
        blocks = []
            
        blocks.append(block(in_channels, out_channels, stride=1))
            
        for i in range(num_blocks-1):
            blocks.append(block(out_channels, out_channels, stride=2))
            
        return nn.Sequential(*blocks)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, x.shape[3])
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
            
        return x 
