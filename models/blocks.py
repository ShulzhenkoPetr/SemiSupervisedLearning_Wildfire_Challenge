from typing import Dict
import torch
from torch import nn


class BasicBlock(nn.Module):

    def __init__(self, in_feats: int, out_feats: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_feats, out_feats, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_feats)
        self.relu1 = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu1(self.bn1(self.conv1(x)))


class ResBlock(nn.Module):

    def __init__(self, in_feats: int, out_feats: int, 
                 downsample: nn.Module = None, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_feats, out_feats, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_feats)
        self.relu1 = nn.ReLU(inplace=False)

        if downsample is not None:
            downsample_sensitive_stride = 2
        else:
            downsample_sensitive_stride = 1
        self.conv2 = nn.Conv2d(out_feats, out_feats, kernel_size=kernel_size, stride=downsample_sensitive_stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_feats)
        
        self.relu2 = nn.ReLU(inplace=False)

        if in_feats != out_feats:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_feats, out_channels=out_feats, kernel_size=1, stride=downsample_sensitive_stride, bias=False),
                nn.BatchNorm2d(out_feats)
            )
        else:
            self.downsample = downsample

    def forward(self, x):
        identity = x

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample:
            identity = self.downsample(identity)

        x += identity
        return self.relu2(x)
    
    

