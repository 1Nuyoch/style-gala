"""
This MobileNetV2 implementation is modified from the following repository:
https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math
from .utils import load_url
from .segm_lib.nn import SynchronizedBatchNorm2d

BatchNorm2d = SynchronizedBatchNorm2d

__all__ = ['mobilenetv2']

model_urls = {
    'mobilenetv2': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/mobilenet_v2.pth.tar',
}


def conv_bn(inp, oup, stride):  # 输入参数：inp 表示输入通道数，oup 表示输出通道数，stride 表示卷积步长。
    return nn.Sequential(  # 返回：一个包含卷积层、批归一化层和 ReLU6 激活函数的序列。
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),  # 使用 nn.Conv2d 定义一个 3x3 的卷积层，步长为 stride，填充为 1，无偏置。
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(  # 返回：一个包含1x1卷积层、批归一化层和ReLU6激活函数的序列。
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):  # expand_ratio：膨胀比例，控制中间层的通道数。
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)  # gen据输入通道数和膨胀比例计算中间隐藏层的通道数
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),  # 深度可分离卷积，输入通道数和输出通道数相同
                BatchNorm2d(hidden_dim),  # 批归一化层。
                nn.ReLU6(inplace=True),  # ReLU6 激活函数
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),  # 逐点卷积，将通道数从 hidden_dim 缩减到 oup。
                BatchNorm2d(oup),  # 批归一化层
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),  # 逐点卷积，将输入通道数缩减到 hidden_dim
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                # 深度可分离卷积，输入和输出通道数都是 hidden_dim
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),  # 逐点卷积，将通道数从 hidden_dim 缩减到 oup
                BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [  # 倒残差块的设置，包括每个块的膨胀比例 t、输出通道数 c、重复次数 n 和步长 s。
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0  # 先确保 input_size 可以被 32 整除，以符合 MobileNetV2 的特性。
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2(pretrained=False, **kwargs):
    """Constructs a MobileNet_V2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV2(n_class=1000, **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['mobilenetv2']), strict=False)
    return model
