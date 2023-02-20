import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels, k=1, mode='expansion'):
        super().__init__()
        
        if mode == 'expansion':
            out_channels = in_channels * k
        else:
            out_channels = in_channels // k
            
        if in_channels == 3:
            out_channels = 64
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        out = self.conv_block(x)
        
        return out

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, k):
        super().__init__()
        out_channels = in_channels // k
        
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, k, mode='reduction')
        
    def forward(self, x, shortcut):
        
        x = self.upsample(x)
        x = torch.cat((x, shortcut), dim=1)
        out = self.conv_block(x)
        
        return out


class Unet(nn.Module):
    def __init__(self, num_classes=151):
        super().__init__()
    
        # encoder
        self.encoder = nn.Sequential(
            ConvBlock(3),
            ConvBlock(64, 2),
            ConvBlock(128, 2),
            ConvBlock(256, 2),
        )
        
        # max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # bottleneck
        self.bottleneck = ConvBlock(512, 2)

        # decoder
        self.decoder = nn.Sequential(
            UpConvBlock(1024, 2),
            UpConvBlock(512, 2),
            UpConvBlock(256, 2),
            UpConvBlock(128, 2)  
        )

        self.final = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
    
    def forward(self, x):
        
        # encoder
        e0 = self.encoder[0](x)
        pool0 = self.pool(e0)
        e1 = self.encoder[1](pool0)
        pool1 = self.pool(e1)
        e2 = self.encoder[2](pool1)
        pool2 = self.pool(e2)
        e3 = self.encoder[3](pool2)
        pool3 = self.pool(e3)
        
        # bottleneck
        b = self.bottleneck(pool3)
        
        # decoder
        d0 = self.decoder[0](b, e3)
        d1 = self.decoder[1](d0, e2)
        d2 = self.decoder[2](d1, e1)
        d3 = self.decoder[3](d2, e0)
        d3 = self.final(d3)
        
        return d3
