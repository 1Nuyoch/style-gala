import torch
import torch.nn as nn
import torch.nn.functional as F


class ECALayer(nn.Module):
    def __init__(self, channel=1, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 一维卷积
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x):
        # 强制将输入转换为 float32 以确保与权重类型一致
        x = x.float()
        y = self.avg_pool(x)  # shape: [batch_size, channels, 1, 1] x:8,512,8,8

        # Reshape to [batch_size, channels, 1] for Conv1d
        y = y.squeeze(-1).transpose(-1, -2)  # shape: [batch_size, channels]

        # Apply Conv1d on the channels
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)  # shape: [batch_size, channels]

        # Restore dimensions: [batch_size, 1, channels] -> [batch_size, channels, 1, 1]
        y = self.sigmoid(y)
        # Element-wise multiplication: [batch_size, channels, height, width] * [batch_size, channels, 1, 1]

        # Interpolate to match the spatial dimensions of `x`
        y = F.interpolate(y, size=(x.shape[2], x.shape[3]), mode="nearest")  # shape: [batch_size, channels, height, width]

        # 元素级相乘后再转回 float16，以继续后续的计算
        return (x * y).to(x.dtype)



