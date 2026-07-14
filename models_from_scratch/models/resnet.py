import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ShortcutIdentityBlock(nn.Module):
    def __init__(self, in_channels, reduction=4, se=False, expansion=4, kernel_size=3, groups=1, channels_per_group=64):
        super().__init__()
        self.shortcut = nn.Identity()

        out_channels = (int(in_channels * channels_per_group / 64) * groups) // reduction
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, groups=groups, padding=1, bias=False),
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
    def __init__(self, in_channels, version, reduction=2, se=False, expansion=4, kernel_size=3, stride=2, groups=1, channels_per_group=64):
        super().__init__()
        
        out_channels = (int(in_channels * channels_per_group / 64) * groups) // reduction
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,  kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels * expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.ReLU()
        )
        
        if version == 'D':
            self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                    nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(in_channels * expansion)
            )
            
        elif version == 'B':
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(in_channels * expansion)
            )
        
        # squeeze excitation block 
        if se:
            self.post_conv = SE_Block(in_channels * expansion)
            
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
    
class ResNet(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3], version='B', se=False, groups=1, channels_per_group=64, num_classes=10):
        super().__init__()
        self.version = version
        self.se = se
        self.groups = groups
        self.channels_per_group = channels_per_group
        
        # zero layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=4, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # 1-4 layers 
        self.layer1 = self._create_layer(layers[0], 64, zero_block=True)
        self.layer2 = self._create_layer(layers[1], 256)
        self.layer3 = self._create_layer(layers[2], 512)
        self.layer4 = self._create_layer(layers[3], 1024)
        
        # linear layer
        self.linear = nn.Linear(2048, num_classes)
    
    def _create_layer(
        self,
        num_blocks,
        in_channels,
        zero_block=False
        ):
        
        blocks = []
        
        if zero_block:
            
            expansion = 4
            
            blocks.append(
                ShortcutConvBlock(in_channels, self.version, expansion=expansion, reduction=1, se=self.se, groups=self.groups, channels_per_group=self.channels_per_group)
            )

            for _ in range(num_blocks - 1):
                blocks.append(
                    ShortcutIdentityBlock(in_channels * expansion, se=self.se, groups=self.groups, channels_per_group=self.channels_per_group)
                )
        
        else:
            
            expansion = 2
            
            blocks.append(
                ShortcutConvBlock(in_channels, self.version, expansion=expansion, reduction=2, se=self.se, groups=self.groups, channels_per_group=self.channels_per_group)
            )

            for _ in range(num_blocks - 1):
                blocks.append(
                    ShortcutIdentityBlock(in_channels * expansion, se=self.se, groups=self.groups, channels_per_group=self.channels_per_group)
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

class ResNext(ResNet):
    def __init__(self, layers=[3, 4, 6, 3], version='B', groups=32, channels_per_group=4):
        super().__init__(layers=layers, version=version, groups=groups, channels_per_group=channels_per_group)
        
class ResNet50(ResNet):
    def __init__(self, layers=[3, 4, 6, 3], version='D', groups=1, channels_per_group=4):
        super().__init__(layers=layers, version=version, groups=groups, channels_per_group=channels_per_group)