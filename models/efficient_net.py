import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MbConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, kernel_size=3, stride=1, se=True, st_depth=True, life_prob=0.8, r=4):
        super().__init__()
        self.stride = stride
        self.use_shortcut = (stride == 1 and in_channels == out_channels)
        
        padding = (kernel_size - 1) // 2
        inter_channels = in_channels * expansion

        # expansion covlolution 
        self.exp_conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU()
        )

        # depthwise convolution 
        self.dwise_conv = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=kernel_size, groups=inter_channels, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU()
        )
        
        # pointwise convolution 
        self.pwise_conv = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # sqeeze and excitation block
        if se:
            self.post_conv = SE_Block(inter_channels)
        else:
            self.post_conv = nn.Identity()
        
        # stochastic depth trick
        if st_depth:
            self.st_depth_block = StochasticDepth(life_prob=life_prob)
        else:
            self.st_depth_block = nn.Identity()
            
    def forward(self, x):
        
        out = self.exp_conv(x)
        out = self.dwise_conv(out)
        out = self.post_conv(out)
        out = self.pwise_conv(out)
        if self.use_shortcut:
            out = self.st_depth_block(out)
            out += x
            
        return out 

class SE_Block(nn.Module):
    def __init__(self, channels, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1) # 1 x 1 x C
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.SiLU(),
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

class StochasticDepth(nn.Module):
    def __init__(self, life_prob=0.8):
        super().__init__()
        self.life_prob = life_prob
        
    def forward(self, x):
        if not self.training:
            b = float(np.random.binomial(n=1, p=self.life_prob))
            x = (b / self.life_prob) * x
        
        return x 

class EfficientNet(nn.Module):
    def __init__(self, width=1., depth=1., dropout=0.2, se=True, st_depth=True, life_prob=0.8, r=4, num_classes=10):
        super().__init__()
        self.in_channels = np.ceil(width * 32).astype(int)
        self.expansions = [1, 6, 6, 6, 6, 6, 6]
        self.out_channels = np.ceil(width * np.array([16, 24, 40, 80, 112, 192, 320])).astype(int).tolist()
        self.num_layers = np.ceil(depth * np.array([1, 2, 2, 3, 3, 4, 1])).astype(int).tolist()
        self.kernels = [3, 3, 5, 3, 5, 5, 3]
        self.strides = [1, 2, 2, 2, 1, 2, 1]
        self.se = se
        self.st_depth = st_depth
        self.life_prob = life_prob # life probability for StochasticDepth trick
        self.r = r # reduction parameter for SE block
        
        final_channels = 1280

        # zero layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.SiLU()
        )

        # 1 - 8 layers
        self.conv_blocks = self._create_layers()

        # 9 layer
        self.last_conv = nn.Sequential(
            nn.Conv2d(self.out_channels[-1], final_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # linear layer
        self.linear = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes)
        )

    def _create_layers(self):
        
        blocks = []
        
        blocks.append(MbConvBlock(self.in_channels, self.out_channels[0],
                                  self.expansions[0], kernel_size=self.kernels[0],
                                  stride=self.strides[0], se=self.se, st_depth=self.st_depth,
                                  life_prob=self.life_prob, r=self.r)
                     )
        self.in_channels = self.out_channels[0]
        blocks += [MbConvBlock(self.in_channels, self.out_channels[0],
                               self.expansions[0], kernel_size=self.kernels[0],
                               stride=1, se=self.se, st_depth=self.st_depth,
                               life_prob=self.life_prob, r=self.r)
                   for _ in range(self.num_layers[0] - 1)
                  ]
        
        for i in range(1, len(self.num_layers)):
            blocks.append(MbConvBlock(self.in_channels, self.out_channels[i],
                                      self.expansions[i], kernel_size=self.kernels[i],
                                      stride=self.strides[i], se=self.se, st_depth=self.st_depth,
                                      life_prob=self.life_prob, r=self.r)
                         )
            self.in_channels = self.out_channels[i]
            blocks += [MbConvBlock(self.in_channels, self.out_channels[i],
                                   self.expansions[i], kernel_size=self.kernels[i],
                                   stride=1, se=self.se, st_depth=self.st_depth,
                                   life_prob=self.life_prob, r=self.r)
                       for _ in range(self.num_layers[i] - 1)
                      ]
           
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv_blocks(x)
        x = self.last_conv(x)
        x = x.view(-1, x.shape[1])
        x = self.linear(x)
        
        return x

class EfficientNetB0(EfficientNet):
    def __init__(self, se=True, st_depth=True, dropout=0.2, life_prob=0.8, r=4):
        super().__init__(se=se, st_depth=st_depth, dropout=dropout, life_prob=life_prob, r=r)

class EfficientNetB2(EfficientNet):
    def __init__(self, width=1.1, depth=1.2, dropout=0.3, se=True, st_depth=True, life_prob=0.8, r=4):
        super().__init__(width=width, depth=depth, dropout=dropout, se=se, st_depth=st_depth, life_prob=life_prob, r=r)

class EfficientNetB7(EfficientNet):
    def __init__(self, width=2., depth=3.1, dropout=0.5, se=True, st_depth=True, life_prob=0.8, r=4):
        super().__init__(width=width, depth=depth, dropout=dropout, se=se, st_depth=st_depth, life_prob=life_prob, r=r)
