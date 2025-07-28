"""Modified from https://github.com/CSAILVision/semantic-segmentation-pytorch"""

import math

import torch.nn as nn
from torch.nn import BatchNorm2d

from .utils import load_url

__all__ = ['ResNet', 'resnet50']

model_urls = {
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # 扩展因子，指示了输出通道数相对于输入通道数的倍数，这里为 4。这是因为在 Bottleneck 结构中，第三个卷积层将输出扩展到 planes * 4。
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # 将输入 x 存储到 residual 变量中，作为残差连接的一部分。
        # 将输入 x 经过第一层卷积、批归一化和激活函数处理后得到 out。
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 将 out 经过第二层卷积、批归一化和激活函数处理后得到 out。
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 将 out 经过第三层卷积和批归一化处理后得到 out，此时 out 的通道数为 planes * 4。
        out = self.conv3(out)
        out = self.bn3(out)
        # 如果定义了 downsample 函数，对 x 进行下采样（跳跃连接中的变换），并将结果存储到 residual 中。
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层，用于降低特征图尺寸。

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)  # 全局平均池化层，用于将最后一个卷积层输出的特征图转换为一维向量。
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层，用于分类任务，将特征向量映射到 num_classes 维度的输出

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 如果 stride 不为 1 或者输入通道数 self.inplanes 与输出通道数 planes * block.expansion 不一致，则定义一个下采样函数 downsample。
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 将输入 x 依次经过 conv1、bn1、relu1、conv2、bn2、relu2、conv3、bn3、relu3 进行特征提取和非线性变换。
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)  # 经过 maxpool 进行下采样。

        x = self.layer1(x)  # 将 x 依次通过 layer1、layer2、layer3、layer4 四个阶段的层。
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # 经过 avgpool 进行全局平均池化，将特征图转换为一维向量。
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # 经过 fc 全连接层，得到模型的输出。

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet18']))
    return model
