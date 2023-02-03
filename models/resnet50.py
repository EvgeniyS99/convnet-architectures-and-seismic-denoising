import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class IdentityBlock(nn.Module):
    def __init__(self, filters, se=False, SE_Block=None, stride=1):
        super().__init__()
        
        self.f1, self.f2, self.f3 = filters
        self.stride = stride
        self.se = se

        
        self.shortcut = nn.Identity()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.f1, self.f2, kernel_size=1, stride=self.stride),
            nn.BatchNorm2d(self.f2),
            nn.ReLU(),
            nn.Conv2d(self.f2, self.f2, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(self.f2),
            nn.ReLU(),
            nn.Conv2d(self.f2, self.f3, kernel_size=1, stride=self.stride),
            nn.BatchNorm2d(self.f3),
            nn.ReLU()
        )
        
        # squeeze excitation block 
        if self.se:
            self.se_block = SE_Block(self.f3)
            
        else:
            self.se_block = nn.Identity()
        
    def forward(self, x):
        #print(x.shape, 'x without conv')
        out = self.conv_block(x)
        out = self.se_block(out)
        #print(out.shape, 'this is out')
        #print(x.shape, 'this is x')
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, filters, version, se=False, SE_Block=None, stride=2, padding=0):
        super().__init__()
        
        self.f1, self.f2, self.f3 = filters
        self.stride = stride
        self.padding = padding
        self.version = version
        self.se = se
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.f1, self.f2, kernel_size=1, stride=self.stride),
            nn.BatchNorm2d(self.f2),
            nn.ReLU(),
            nn.Conv2d(self.f2, self.f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.f2),
            nn.ReLU(),
            nn.Conv2d(self.f2, self.f3, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.f3),
            nn.ReLU()
        )
        
        if self.version == 'A':
            self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=self.stride, padding=self.padding),
                    nn.Conv2d(self.f1, self.f3, kernel_size=1, stride=1),
                    nn.BatchNorm2d(self.f3)
            )
            
        elif self.version == 'B':
            self.shortcut = nn.Sequential(
                    nn.Conv2d(self.f1, self.f3, kernel_size=1, stride=self.stride),
                    nn.BatchNorm2d(self.f3)
            )
        
        # squeeze excitation block 
        if self.se:
            self.se_block = SE_Block(self.f3)
            
        else:
            self.se_block = nn.Identity()
            
            
    def forward(self, x):
        #print(x.shape, 'x without conv')    
        out = self.conv_block(x)
        out = self.se_block(out)
        #print(out.shape, 'this is out')
        #print(self.shortcut(x).shape, 'this is x ')
        out += self.shortcut(x)
        out = F.relu(out)
            
        return out
    
class SE_Block(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1) # 1 x 1 x C
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(),
            nn.Linear(channels // r, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        N, C, _, _ = x.shape
        out = self.squeeze(x)
        #print(out.shape, 'after squeeze')
        out = out.view(-1, C)
        #print(out.shape, 'after view')
        out = self.excitation(out)
        #print(out.shape, 'after excitation')
        out = out.view(N, C, 1, 1)
        out = out * x
        
        return out
    
class ResNet50(nn.Module):
    def __init__(self, conv_block, identity_block, version, se=False, SE_Block=None, num_classes=10):
        super().__init__()
        self.version = version
        self.se = se
        
        # zero layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # first layer
        self.conv_block1 = conv_block([64, 64, 256], self.version, se=self.se, SE_Block=SE_Block, stride=1, padding=1)
        self.identity_block1 = identity_block([256, 64, 256], se=self.se, SE_Block=SE_Block)
        self.identity_block2 = identity_block([256, 64, 256], se=self.se, SE_Block=SE_Block)
        
        # second layer
        self.conv_block2 = conv_block([256, 128, 512], self.version, se=self.se, SE_Block=SE_Block, stride=2, padding=1)
        self.identity_block3 = identity_block([512, 128, 512], se=self.se, SE_Block=SE_Block)
        self.identity_block4 = identity_block([512, 128, 512], se=self.se, SE_Block=SE_Block)
        self.identity_block5 = identity_block([512, 128, 512], se=self.se, SE_Block=SE_Block)
        
        # third layer
        self.conv_block3 = conv_block([512, 256, 1024], self.version, se=self.se, SE_Block=SE_Block, stride=2, padding=1)
        self.identity_block6 = identity_block([1024, 256, 1024], se=self.se, SE_Block=SE_Block)
        self.identity_block7 = identity_block([1024, 256, 1024], se=self.se, SE_Block=SE_Block)
        self.identity_block8 = identity_block([1024, 256, 1024], se=self.se, SE_Block=SE_Block)
        self.identity_block9 = identity_block([1024, 256, 1024], se=self.se, SE_Block=SE_Block)
        self.identity_block10 = identity_block([1024, 256, 1024], se=self.se, SE_Block=SE_Block)
        
        # fourth layer 
        self.conv_block4 = conv_block([1024, 512, 2048], self.version, se=self.se, SE_Block=SE_Block, stride=2, padding=1)
        self.identity_block11 = identity_block([2048, 512, 2048], se=self.se, SE_Block=SE_Block)
        self.identity_block12 = identity_block([2048, 512, 2048], se=self.se, SE_Block=SE_Block)
        
        # linear layer
        self.linear = nn.Linear(2048, num_classes)
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.identity_block2(self.identity_block1(self.conv_block1(x)))
        
        x = self.identity_block5(self.identity_block4(self.identity_block3(self.conv_block2(x))))
        
        x = self.identity_block10(self.identity_block9(self.identity_block8(self.identity_block7(self.identity_block6(self.conv_block3(x))))))
        
        x = self.identity_block12(self.identity_block11(self.conv_block4(x)))
        
        x = F.avg_pool2d(x, x.shape[3])
        #print(x.shape, 'final shape')
        x = x.view(x.shape[0], -1)
        #print(x.shape, 'prelinear shape')
        x = self.linear(x)
        
        return x

