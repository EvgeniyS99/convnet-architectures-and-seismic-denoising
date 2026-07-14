import torch
import torch.nn as nn
import numpy as np


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), # 96 x 55 x 55
            nn.ReLU(),
            nn.LocalResponseNorm(k=2, size=5, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2) # 96 x 27 x 27
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), # 256 x 27 x 27
            nn.ReLU(),
            nn.LocalResponseNorm(k=2, size=5, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2) # 256 x 13 x 13 
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), # 384 x 13 x 13
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), # 384 x 13 x 13
            nn.ReLU()
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), # 384 x 13 x 13
            nn.ReLU()
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), # 256 x 13 x 13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2) # 256 x 6 x 6 
        )
        
        self.clf = nn.Sequential(
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )
        
    def forward(self, x):
        """Forward pass"""
        x = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x))))))
        x = x.view(-1, 256 * 6 * 6) # squeeze the output tensor 
        x = self.clf(x)
            
        return x 
