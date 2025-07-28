# Modified from https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import rotate
import torch.fft as fft
from icecream import ic
import PIL
#将特征图 (feats) 转换成图像网格，并保存为一个 PNG 文件。这个函数常用于可视化神经网络的中间层输出。
#feats: 输入的特征图，是一个张量 (torch.Tensor)。fname: 保存图像的文件名。gridsize: 图像网格的大小，元组形式 (gw, gh)，其中 gw 是网格的宽度（列数），gh 是网格的高度（行数）。
def save_image_grid(feats, fname, gridsize):
    gw, gh = gridsize#解包网格的宽度和高度
    idx = gw * gh#计算出网格中最大图片数量
#表示特征图中的最大值和最小值。
    max_num = torch.max(feats[:idx]).item()
    min_num = torch.min(feats[:idx]).item()
    feats = feats[:idx].cpu() * 255 / (max_num - min_num) #将特征图归一化到 0-255 之间，并转换为 CPU 张量。
    feats = np.asarray(feats, dtype=np.float32)#将张量转换为 NumPy 数组。
    feats = np.rint(feats).clip(0, 255).astype(np.uint8)#对特征图进行四舍五入，并裁剪到 0-255 的范围内，转换为 8 位无符号整数。

    C, H, W = feats.shape
#将特征图重塑为 (gh, gw, 1, H, W) 形状，其中 gh 和 gw 分别是行数和列数，H 和 W 是每个特征图的高度和宽度。
    feats = feats.reshape(gh, gw, 1, H, W)
    feats = feats.transpose(0, 3, 1, 4, 2)#交换数组的维度顺序，使其适合保存为图像。
    feats = feats.reshape(gh * H, gw * W, 1)#将图像展开成二维图像
    feats = np.stack([feats]*3, axis=2).squeeze() * 10#将单通道的特征图复制三次，生成一个三通道的灰度图像。乘以 10 以增强图像的对比度。
    feats = np.rint(feats).clip(0, 255).astype(np.uint8)
    
    from icecream import ic
    ic(feats.shape)#调试语句，使用 icecream 库打印特征图的形状。
    
    feats = PIL.Image.fromarray(feats)#将 NumPy 数组转换为 PIL 图像对象
    feats.save(fname + '.png')#保存图像为 PNG 格式。
#卷积操作函数，基于 PyTorch 的 torch.nn.functional.conv2d 实现。
#input: 输入的张量，通常是一个图像或特征图。weight: 卷积核的权重张量。bias: 卷积核的偏置项（可选）。stride: 卷积操作的步幅，默认为 1。
# padding: 卷积操作的填充大小，默认为 0。dilation: 卷积操作的膨胀系数，默认为 1。groups: 分组卷积的组数，默认为 1
def _conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv2d(input=input, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
#使用 F.conv2d 直接调用 PyTorch 的二维卷积函数。这个函数将 input 与 weight 进行卷积，得到输出张量。

'''
神经网络模块 LeSpTrWr，用于在输入数据上应用可学习的空间变换（例如旋转），然后应用一个内部模型 (impl)，最后再反向应用变换以恢复输出的原始空间布局。
mpl: 这是被包装的模型或函数，即将在空间变换后应用于输入数据的模型。
pad_coef: 用于控制在进行旋转之前对输入数据进行填充的系数。填充可以避免旋转后图像边缘的信息丢失。
angle_init_range: 初始化角度的范围。self.angle 是一个随机角度，范围在 0 到 angle_init_range 之间。
train_angle: 指定是否将旋转角度 self.angle 作为可训练参数。如果 train_angle=True，则 self.angle 会变成一个 nn.Parameter，并在训练过程中优化。
'''
class LearnableSpatialTransformWrapper(nn.Module):
    def __init__(self, impl, pad_coef=0.5, angle_init_range=80, train_angle=True):
        super().__init__()
        self.impl = impl
        self.angle = torch.rand(1) * angle_init_range
        if train_angle:
            self.angle = nn.Parameter(self.angle, requires_grad=True)
        self.pad_coef = pad_coef

    def forward(self, x):
        if torch.is_tensor(x):
# x 应用空间变换 (self.transform)。将变换后的结果传入 impl 模型进行处理。对处理后的结果应用逆变换 (self.inverse_transform) 以恢复其原始空间布局
            return self.inverse_transform(self.impl(self.transform(x)), x)
        elif isinstance(x, tuple):
#对元组中的每个元素分别应用空间变换 (self.transform)。将变换后的元组传入 impl 进行处理。对处理后的每个元素分别应用逆变换 (self.inverse_transform)。
            x_trans = tuple(self.transform(elem) for elem in x)
            y_trans = self.impl(x_trans)
            return tuple(self.inverse_transform(elem, orig_x) for elem, orig_x in zip(y_trans, x))
        else:
            raise ValueError(f'Unexpected input type {type(x)}')

#获取输入张量的高度 (height) 和宽度 (width)。根据 pad_coef 计算需要的填充量 pad_h 和 pad_w。
# 使用F.pad函数在输入张量的边缘添加填充，填充模式为“反射”（reflect），即边缘像素反射到填充区域中。使用rotate函数将填充后的张量旋转self.angle指定的角度
# 返回旋转后的张量
    def transform(self, x):
        height, width = x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
        x_padded = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode='reflect')
        x_padded_rotated = rotate(x_padded, angle=self.angle.to(x_padded))
        return x_padded_rotated

    def inverse_transform(self, y_padded_rotated, orig_x):
        height, width = orig_x.shape[2:]#获取原始输入张量的高度和宽度。
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)#计算所需的填充量 pad_h 和 pad_w。

        y_padded = rotate(y_padded_rotated, angle=-self.angle.to(y_padded_rotated))#使用与变换相反的角度-self.angle将y_padded_rotated旋转回去
        y_height, y_width = y_padded.shape[2:]#就会返回一个包含高度和宽度的元组
        #通过切片操作从填充后的张量 y_padded 中提取出中心区域，去掉填充部分，恢复到原始输入图像的尺寸
        #高度维度上，从 pad_h 开始到 y_height - pad_h，即去掉上下的填充部分。宽度维度上，从 pad_w 开始到 y_width - pad_w，即去掉左右的填充部分。
        y = y_padded[:, :, pad_h : y_height - pad_h, pad_w : y_width - pad_w]
        return y


class SELayer(nn.Module):#通过自适应地为每个通道分配权重来增强神经网络的表达能力
    def __init__(self, channel, reduction=16):#channel：表示输入特征图的通道数。reduction：是通道数的压缩比，默认为 16。
                        # 通过 channel // reduction，在全连接层中将通道数减少，随后再恢复到原始通道数。这样做的目的是减少参数量和计算量。
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#将输入特征图池化为大小为 1x1 的特征图。每个通道会有一个单独的全局平均值。
        #self.fc 是一个全连接层序列。首先通过 nn.Linear(channel, channel // reduction, bias=False) 将通道数减少到原来的 1/reduction
        #通过 nn.ReLU 激活，再通过第二个nn.Linear层将通道数恢复到原始通道数，最后通过nn.Sigmoid将输出限制在0到1之间。这个输出会作为每个通道的权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()#获取输入 x 的批次大小 b 和通道数 c，这里 _ 表示空间维度（高度和宽度），但后续操作中不需要具体知道它们的值。
        y = self.avg_pool(x).view(b, c)#通过avg_pool操作将输入x池化到每个通道的平均值,输出尺寸为(b,c,1,1)。再通过view(b, c)将其变形为(b,c)，即每个样本有c个通道
        y = self.fc(y).view(b, c, 1, 1)#经过全连接层self.fc，y的尺寸从(b,c)变为(b,c)（注意这里的通道数是先压缩再恢复的,再通过view(b,c,1,1)变形为(b,c,1,1)，为广播乘法做准备
        res = x * y.expand_as(x)#将 y的尺寸扩展为与输入 x 相同的 (b, c, H, W)。然后通过逐元素相乘，x 的每个通道特征图都会被相应的权重 y调整
        return res #返回加权后的特征图 res


class FourierUnit(nn.Module):
#神经网络模块，它将输入特征图转换为频域表示（通过傅里叶变换），对频域特征进行处理，再将其转换回空间域。被用于处理和增强特征的频率成分，改善模型对复杂特征的捕捉能力
    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
#输入和输出的通道数。groups：用于分组卷积，默认为1，不分组。用于调整特征图的空间尺度，默认为None，不调整。如果设置了spatial_scale_factor，特征图会被缩放，缩放模式由spatial_scale_mode 决定（如 'bilinear' 插值
#布尔值，决定是否在频域中添加位置编码。use_se：布尔值，决定是否使用Squeeze-and-Excitation模块。ffc3d：布尔值，决定是否使用3D快速傅里叶变换 (FFT)。fft_norm：控制 FFT 规范化的参数，默认是 'ortho'，表示正交规范化
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups#将传入的 groups 参数赋值给类属性 self.groups。groups 用于指定卷积层的分组数量
#定义了一个 2D 卷积层,输入通道数，由 in_channels * 2 和 spectral_pos_encoding 决定。如果使用了 spectral_pos_encoding，输入通道数会增加 2
#输出通道数，设置为 out_channels * 2。倍增的原因可能是为了同时处理实部和虚部（例如，在频域中处理复数）。
#kernel_size=1：使用 1x1 的卷积核，通常用于改变特征图的通道数，而不改变特征图的空间尺寸。stride=1：卷积的步幅为 1，意味着卷积操作不会跳过输入的任何部分。
#padding=0：没有填充，输出的空间尺寸会小于或等于输入的尺寸。groups=self.groups：指定卷积的分组数，默认为 1，即不分组。bias=False：卷积层没有偏置项
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.relu = torch.nn.ReLU(inplace=False)
#定义了一个 ReLU 激活函数，用于在卷积之后进行非线性变换。inplace=False 表示激活函数不会直接修改输入数据，而是会生成一个新的输出张量。
        # 添加squeeze and excitation block
        self.use_se = use_se#将 use_se 参数的值赋给类属性 self.use_se。
        if use_se:#则初始化 SE 块。
            if se_kwargs is None:
                se_kwargs = {}#默认为一个空字典。se_kwargs 用于传递额外的参数给 SELayer。
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)#创建一个SELayer实例，输入通道数是self.conv_layer.in_channels，并传递se_kwargs中的其他参数

        self.spatial_scale_factor = spatial_scale_factor#空间缩放因子，用于调整输入特征图的尺寸。
        self.spatial_scale_mode = spatial_scale_mode#空间缩放模式，例如双线性插值 ('bilinear') 等
        self.spectral_pos_encoding = spectral_pos_encoding#布尔值，决定是否在频域中添加位置编码。
        self.ffc3d = ffc3d#布尔值，决定是否使用 3D 快速傅里叶变换 (FFT)。
        self.fft_norm = fft_norm#FFT 规范化的类型，默认是 'ortho'。

    def forward(self, x):
        batch = x.shape[0]#获取输入 x 的批次大小。

#如果指定了 spatial_scale_factor，输入 x 会按比例缩放。
        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)
#通过插值操作改变输入特征图 x 的空间尺寸，缩放的倍数由 self.spatial_scale_factor 决定，插值方式由 self.spatial_scale_mode 决定，align_corners=False 则确保了插值过程的自然性和一致性。
#对输入 x 进行傅里叶变换（FFT），并将结果拆分为实部和虚部。这些频域特征将被重新排列，并转化为适合卷积处理的格式。
        r_size = x.size()#获取输入张量 x 的尺寸，并将其存储在变量r_size 中。r_size是一个元组，通常是 (batch_size, channels, height, width)。
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)#傅里叶变换要作用的维度
        ffted = fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)#对输入 x 进行多维快速傅里叶变换，计算结果为频域表示
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)#将实部和虚部分别堆叠在一起，沿着新增加的最后一维dim=-1形成一个新的张量。将复数表示转换为两个独立的实数通道
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)改变张量的维度顺序。将张量从形状 (batch, c, h, w/2+1, 2) 重新排列为 (batch, c, 2, h, w/2+1)。
        #为了让张量符合后续卷积操作的输入格式。contiguous：确保重新排列后的张量在内存中是连续的。因为只有连续的张量才能正确地改变形状。
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
#将张量的形状改变为 (batch, -1, h, w/2+1)，-1 表示让PyTorch自动计算该维度的大小，以确保总元素数保持不变。将实部和虚部压缩到一个通道维度中
        #对输入的张量 x 进行傅里叶变换，将其转换到频域后，在频域上进行卷积操作，再通过逆傅里叶变换将处理后的频域特征图还原为时域特征图
        if self.spectral_pos_encoding:#频谱位置编码，为傅里叶变换的结果添加位置编码。
            height, width = ffted.shape[-2:]#获取当前特征图的高度和宽度；生成从0到1的线性序列，用于表示垂直方向的位置编码， 扩展张量的维度以匹配批量和特征图的大小
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)#将生成的垂直和水平位置编码与频域特征拼接在一起，形成扩展后的特征图
#位置编码被附加到频域特征中，使得后续的卷积操作能够感知图像的位置信息。
        if self.use_se:#则将频域特征输入到SE模块中。SE模块通过全局平均池化和全连接层来捕获通道间的关系，并通过自适应权重重新调整特征图的通道权重。
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)，1x1卷积层，用于在频域上处理特征图。输出特征图的通道数为输入通道数的两倍
        ffted = self.relu(ffted)#对卷积输出应用ReLU激活函数，增加非线性
#先将通道重新组织为复数的实部和虚部。通过 torch.complex 将实部和虚部重新组合为复数形式的张量。
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
#对频域特征进行逆傅里叶变换，恢复到时域表示；slice 确定了逆变换的目标形状。如果处理的是3D数据（self.ffc3d=True），则逆变换考虑最后三个维度，否则考虑最后两个维度
        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output

#
class SpectralTransform(nn.Module):
#控制分组卷积的分组数量；是否启用局部频域单元（LFU，Local Fourier Unit）
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:#输入图像会在处理前进行下采样；使用2x2的平均池化
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(#1x1卷积层，通道数减少到原始的一半，接着ReLU激活函数。这个卷积层对输入进行通道降维处理，为后续的频域处理准备数据。
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            # nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(#定义一个 FourierUnit（傅里叶单元），它对经过 self.conv1 处理后的特征图在频域上进行进一步的处理。
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:#如果启用了局部频域单元（LFU），则定义另一个 FourierUnit，用于局部频域处理。
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(# 一个1x1卷积层，用于将处理后的特征图恢复到预定的输出通道数。
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)#如果需要，下采样输入。
        x = self.conv1(x)# 对输入应用第一个卷积层和ReLU激活
        output = self.fu(x)#使用傅里叶单元对特征图进行频域处理

        if self.enable_lfu:
            n, c, h, w = x.shape#获取特征图的批次、通道、高度和宽度
            split_no = 2#设置分割数量和分割尺寸，这里将特征图的高度和宽度各分为两部分。
            split_s = h // split_no
            xs = torch.cat(torch.split(#将特征图按照高度和宽度分割并重新组合，使得每个局部区域都可以单独处理。
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)#将分割后的特征图输入到 LFU 单元进行局部频域处理
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()#将局部处理后的特征图按原始图像的分割方式重复。
        else:
            xs = 0
#首先将原始卷积后的特征图、全局频域处理后的特征图以及局部频域处理后的特征图进行融合。然后对融合后的特征图进行最后一次卷积，使其符合输出通道数，并返回最终的结果。
        output = self.conv2(x + output + xs)

        return output

#结合了卷积操作和频域变换来处理输入数据。该模块允许在局部（Local）和全局（Global）范围内分别处理不同的输入通道，并通过调整比例参数来控制局部和全局通道的数量
class FFC(nn.Module):
#卷积核的大小；输入通道中用于全局处理（Global）的比例；输出通道中用于全局处理的比例；padding: 卷积操作的填充方式；卷积操作的扩张率；控制分组卷积的分组数量；
#是否在卷积操作中使用偏置；是否启用局部频域单元（LFU）；填充类型（如'reflect'、'zeros'等；是否启用门控机制；用于传递给 SpectralTransform 类的额外参数
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)#输入通道中用于全局处理（in_cg）和局部处理（in_cl）的通道数量。它们分别由 ratio_gin 控制
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)#输出通道中用于全局处理（out_cg）和局部处理（out_cl）的通道数量。它们分别由 ratio_gout 控制
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg
#这个卷积层处理输入的局部通道并输出局部通道。nn.Identity 在输入或输出的局部通道数为0时被用作占位符，相当于“跳过”该卷积层
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d#这个卷积层处理输入的局部通道并输出全局通道
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d# 这个卷积层处理输入的全局通道并输出局部通道
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform#用于处理输入的全局通道并输出全局通道。可能会频域变换，取决于是否启用SpT
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated#如果 gated 为真，并且存在全局或局部通道，则会添加一个卷积层用于门控操作。否则，使用 nn.Identity 占位符
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x, fname=None):
        x_l, x_g = x if type(x) is tuple else (x, 0)#局部分量（local component）和全局分量（global component，如果x是元组，将第一个元素分给x_l，第二个分给x_g。如果x不是元组，则假设只有局部分量，x_l设为x，x_g设为 0
        out_xl, out_xg = 0, 0#输出局部和全局分量，初始值为 0。

        if self.gated:#将局部和全局分量拼接在一起（total_input），然后通过门控卷积层 self.gate 计算门控信号。结果通过 sigmoid 函数归一化到 [0, 1] 范围内。
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)#生成两个门控信号，用于控制全局分量对局部分量的影响(g2l_gate)和局部分量对全局分量的影响(l2g_gate
        else:
            g2l_gate, l2g_gate = 1, 1
            
        # for i in range(x_g.shape[0]):
        #     c, h, w = x_g[i].shape
        #     gh = 3
        #     gw = 3
        #     save_image_grid(x_g[i].detach(), f'vis/{fname}_xg_{h}', (gh, gw))
        
        # for i in range(x_l.shape[0]):
        #     c, h, w = x_l[i].shape
        #     gh = 3
        #     gw = 3
        #     save_image_grid(x_l[i].detach(), f'vis/{fname}_xl_{h}', (gh, gw))
            
        spec_x = self.convg2g(x_g)#将全局分量 x_g 通过频谱变换层 self.convg2g 处理，得到变换后的结果 spec_x
        
        # for i in range(spec_x.shape[0]):
        #     c, h, w = spec_x[i].shape
        #     gh = 3
        #     gw = 3
        #     save_image_grid(spec_x[i].detach(), f'vis/{fname}_spec_x_{h}', (gh, gw))

        if self.ratio_gout != 1:#out_xl是局部分量的输出，由self.convl2l(x_l)处理局部分量，并将全局分量通过self.convg2l(x_g)处理后的结果乘门控信号g2l_gate
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:#out_xg是全局分量的输出，由局部分量通过self.convl2g(x_l)处理后的结果乘以门控信号l2g_gate，再加前面得到的频谱变换结果spec_x
            out_xg = self.convl2g(x_l) * l2g_gate + spec_x
        
        # for i in range(out_xg.shape[0]):
        #     c, h, w = out_xg[i].shape
        #     gh = 3
        #     gw = 3
        #     save_image_grid(out_xg[i].detach(), f'vis/{fname}_outg_{h}', (gh, gw))
        
        # for i in range(out_xl.shape[0]):
        #     c, h, w = out_xl[i].shape
        #     gh = 3
        #     gw = 3
        #     save_image_grid(out_xl[i].detach(), f'vis/{fname}_outl_{h}', (gh, gw))

        return out_xl, out_xg

class FFC_BN_ACT(nn.Module):
#卷积核大小；输入和输出的全局特征比例；卷积参数，分别表示步幅、填充、扩张以及分组；卷积层是否使用偏置；用于批量归一化的层，默认为 nn.Sy；用于激活的层，默认为 nn.Ide
#填充类型，默认为 reflect 填充。enable_lfu：是否启用局部频域单元（Local Fourier Unit，LFU），即是否对局部特征进行频域变换。kwargs：传递给 FFC 的其他参数。
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.SyncBatchNorm, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()#创建实例
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
#根据 ratio_gout 的值决定是否应用批量归一化。如果输出全部是局部特征或全局特征，则直接使用 nn.Identity 作为归一化层，即不执行归一化
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        # self.bn_l = lnorm(out_channels - global_channels)
        # self.bn_g = gnorm(global_channels)
#根据 ratio_gout 的值决定是否应用激活函数。若不需要激活，则使用 nn.Identity 作为占位符。
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x, fname=None):
        x_l, x_g = self.ffc(x, fname=fname,)#输入 x 通过 FFC 模块处理，得到局部和全局特征 x_l 和 x_g
        x_l = self.act_l(x_l)#对局部特征 x_l 和全局特征 x_g 分别应用激活函数（如果设置了激活层）
        x_g = self.act_g(x_g)
        return x_l, x_g


class FFCResnetBlock(nn.Module):#通过 FFC_BN_ACT 模块来实现局部和全局特征的分离和处理，并且通过残差连接来保留输入信息
    #输入和输出特征的通道数；填充类型（例如 'reflect'、'replicate'），用于卷积操作；用于批量归一化的层；用于激活的层，默认nn.ReLU；膨胀卷积的膨胀系数，默认1
    #可选的参数，用于空间变换的额外配置；布尔值，决定是否将局部和全局特征合并为一个输出；控制全局特征的输入和输出通道比例，默认均为 0.75
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, ratio_gin=0.75, ratio_gout=0.75):
        super().__init__()
        #初始化，conv1 和 conv2 是两个 FFC_BN_ACT 模块，用于进行两次卷积操作。这些卷积层会处理输入的局部和全局特征。
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout)
        if spatial_transform_kwargs is not None:#使用 LearnableSpatialTransformWrapper 包裹 conv1 和 conv2，以实现可学习的空间变换
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline#inline 参数决定在前向传播中是否将局部和全局特征拼接成一个张量

    def forward(self, x, fname=None):
        if self.inline:#输入张量 x 被切分为局部部分 x_l 和全局部分 x_g
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:#输入 x 可能已经是一个元组 (x_l, x_g)，否则全局部分 x_g 设置为 0
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g#保存原始的局部和全局特征，后续会用于残差连接
#对局部和全局特征分别通过 conv1 和 conv2 进行两次卷积操作
        x_l, x_g = self.conv1((x_l, x_g), fname=fname)
        x_l, x_g = self.conv2((x_l, x_g), fname=fname)

        x_l, x_g = id_l + x_l, id_g + x_g#执行残差连接，将输入的局部和全局特征加到卷积输出上
        out = x_l, x_g
        if self.inline:#则将局部和全局特征拼接成一个张量作为输出，否则输出 (x_l, x_g) 元组。
            out = torch.cat(out, dim=1)
        return out

class ConcatTupleLayer(nn.Module):#将输入元组中的两个张量（通常代表局部特征和全局特征）按通道维度进行拼接，返回一个拼接后的张量,继承nn.Module，是一个神经网络模块
    def forward(self, x):
        assert isinstance(x, tuple)#确保输入 x 是一个元组（tuple），其中包含局部特征 x_l 和全局特征 x_g。如果输入不是元组，代码会报错
        x_l, x_g = x#将元组中的两个元素解包，分别赋值给 x_l 和 x_g。x_l 通常表示局部特征，x_g 通常表示全局特征
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)#确保局部特征 x_l 或全局特征 x_g 至少有一个是 PyTorch 张量（tensor）。如果都不是张量，代码会报错
        if not torch.is_tensor(x_g):#如果 x_g 不是张量（例如可能是 0 或 None），则直接返回局部特征 x_l，因为此时没有全局特征可拼接
            return x_l
        return torch.cat(x, dim=1)
#如果 x_g 是张量，代码将局部特征 x_l 和全局特征 x_g 沿着通道维度（即 dim=1）进行拼接，返回拼接后的结果