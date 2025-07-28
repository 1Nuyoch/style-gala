
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F  # 如果有使用 F.relu 之类的函数
import torch.optim as optim  # 如果有优化器


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, alpha=0.5):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_channels, bias=False)  # Depthwise
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # Pointwise
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.activation = nn.SiLU()  # Swish 激活函数2+
        # 残差连接1：如果 in_channels == out_channels，直接加和，否则用 1x1 卷积调整维度
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  bias=False) if in_channels != out_channels else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32, requires_grad=True)) #
    def forward(self, x):
        # x_id = x.clone()
        residual = self.residual(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        # return x + x_id  # 残差连接1
        return self.alpha * x + (1 - self.alpha) * residual   #Swish 激活函数2+