import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ShortcutIdentityBlock(nn.Module):
    def __init__(self, in_channels, reduction, se=False, shortcut_identity_expansion=4, kernel_size=3):
        super().__init__()
        self.shortcut = nn.Identity()
        
        out_channels = in_channels // reduction
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        # squeeze excitation block 
        if se:
            self.post_conv = SE_Block(in_channels)
            
        else:
            self.post_conv = nn.Identity()
        
    def forward(self, x):

        out = self.conv_block(x)
        out = self.post_conv(out)
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out
    
class ShortcutConvBlock(nn.Module):
    def __init__(self, in_channels, reduction, version, se=False, shortcut_conv_expansion=4, kernel_size=3, stride=2, padding=0):
        super().__init__()
        out_channels = in_channels // reduction
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,  kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * shortcut_conv_expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * shortcut_conv_expansion),
            nn.ReLU()
        )
        
        if version == 'D':
            self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=stride, padding=padding),
                    nn.Conv2d(in_channels, out_channels * shortcut_conv_expansion, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_channels  * shortcut_conv_expansion)
            )
            
        elif version == 'B':
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * shortcut_conv_expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels  * shortcut_conv_expansion)
            )
        
        # squeeze excitation block 
        if se:
            self.post_conv = SE_Block(out_channels * shortcut_conv_expansion)
            
        else:
            self.post_conv = nn.Identity()
            
    def forward(self, x):
        
        out = self.conv_block(x)
        out = self.post_conv(out)
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
        out = out.view(-1, C)
        out = self.excitation(out)
        out = out.view(N, C, 1, 1)
        out = out * x
        
        return out
    
class ResNet50(nn.Module):
    def __init__(self, shortcut_conv_block, shortcut_identity_block, version, layers, se=False, num_classes=10):
        super().__init__()
        self.version = version
        self.se = se
        
        # zero layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=4, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # 1-4 layers 
        self.layer1 = self._create_layer(shortcut_conv_block, shortcut_identity_block, layers[0], 64, 1, 4, 4, 4)
        self.layer2 = self._create_layer(shortcut_conv_block, shortcut_identity_block, layers[1], 256, 2, 4, 4, 2)
        self.layer3 = self._create_layer(shortcut_conv_block, shortcut_identity_block, layers[2], 512, 2, 4, 4, 2)
        self.layer4 = self._create_layer(shortcut_conv_block, shortcut_identity_block, layers[3], 1024, 2, 4, 4, 2)
        
        # linear layer
        self.linear = nn.Linear(2048, num_classes)
    
    def _create_layer(
        self,
        shortcut_conv_block,
        shortcut_identity_block,
        num_blocks,
        in_channels,
        shortcut_conv_reduction,
        shortcut_identity_reduction,
        shortcut_conv_expansion,
        shortcut_identity_expansion
        ):
        
        blocks = []
        
        blocks.append(
            shortcut_conv_block(in_channels, shortcut_conv_reduction, self.version, se=self.se, shortcut_conv_expansion=shortcut_conv_expansion)
        )
        
        for _ in range(num_blocks - 1):
            blocks.append(
                shortcut_identity_block(in_channels * shortcut_identity_expansion, shortcut_identity_reduction, se=self.se, shortcut_identity_expansion=shortcut_identity_expansion)
            )
            
        return nn.Sequential(*blocks)
            
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = F.avg_pool2d(x, x.shape[3])
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        
        return x