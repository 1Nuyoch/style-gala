import numpy as np
from numpy.lib.type_check import imag
import torch
import torch.nn as nn
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
from icecream import ic
import torch.nn.functional as F
from training.ffc import FFCResnetBlock, ConcatTupleLayer


from training.DSC import DepthwiseSeparableConv
from training.ECA import ECALayer
from training.mdta import MDTA  # 导入 MDTA 模块


# ----------------------------------------------------------------------------
'''
class ReconstructionModule(nn.Module):
    def __init__(self, img_channels):
        super(ReconstructionModule, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, img_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.conv3(x)
        return x
'''

@misc.profiled_function
# 对张量 x 进行二阶矩归一化，确保其在指定维度上的平方均值为 1。具体来说，它通过调整张量的幅度（大小）来使其具有均匀的分布
def normalize_2nd_moment(x, dim=1, eps=1e-8):  # dim: 要进行归一化的维度，默认为 1（通常是通道维度;eps: 一个小常数，用来防止除以零的情况
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()  # 首先对张量 x 的每个元素进行平方计算，沿着指定的 dim 维度计算平方后的均值。


# keepdim=True 保证计算后的张量保持原来的维度数，避免降维；为了防止后续除以零的情况，添加一个非常小的常数eps，通过.rsqrt() 计算平方根的倒数（逆平方根）。
# 这相当于 1 / sqrt(...)，可以用于缩放原始张量，最后，将原始张量 x 与计算出来的逆平方根相乘。这一步将调整 x 的幅度，使得它的平方均值为 1，即二阶矩被归一化
@misc.profiled_function
def modulated_conv2d(
        x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
        weight,  # Weight卷积核权重张量 tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
        styles,  # Modulation coefficients 风格调制系数 of shape [batch_size, in_channels].
        noise=None,  # Optional noise tensor to add to the output activations.在卷积结果上增加噪声
        up=1,  # Integer upsampling factor.上采样因子。默认值为 1（不进行上采样
        down=1,  # Integer downsampling factor.下采样因子。默认值为 1（不进行下采样
        padding=0,  # Padding with respect to the upsampled image.卷积操作中的填充，保证卷积不改变图像尺寸
        resample_filter=None,
        # 低通滤波器 to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
        demodulate=True,  # Apply weight demodulation?是否在卷积过程中应用去调制操作。去调制是为了避免特征图的放大或缩小不均衡。
        flip_weight=True,
        # False = convolution, True = correlation (matches torch.nn.functional.conv2d).控制卷积类型，Fal标准卷积，Tr相关操作
        fused_modconv=True,
        # Perform modulation, convolution, and demodulation as a single fused operation?是否将调制、卷积和去调制作为一个单独的操作来执行
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape  # 确保输入的张量 weight、x 和 styles 具有正确的形状
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.如果输入数据类型是 float16，为了避免数值过大导致溢出，对 weight 和 styles 进行归一化
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1, 2, 3],
                                                                            keepdim=True))  # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.分别用于存储调制后的权重和去调制系数
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]将卷积核的权重扩展一个批次维度，并根据样本的风格系数对权重进行调制
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]其中 N 是 batch_size，k：表示 kernel size，即卷积核的大小
    # 使形状从[out_channels, in_channels, kernel_height, kernel_width]变[1, out_channels, in_channels, kh, kw]，1代表batch维度
    # 使用风格调制系数 styles 对权重进行逐样本的调制。styles 是一个形状为 [batch_size, in_channels] 的张量，表示每个样本每个输入通道的风格调制系数
    if demodulate:  # 启用了去调制（demodulate=True），则计算 dcoefs，即去调制系数。去调制的目的是消除每个样本的权重过大或过小的影响，防止不平衡
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]对权重的平方和求和后，加上一个小的数值1e-8，计算平方和的平方根的倒数rsqrt，得到去调制系数
    if demodulate and fused_modconv:  # 去调制和融合卷积操作都启用，根据去调制系数 dcoefs 对调制后的权重进行缩放。
        # dcoefs 的形状为 [batch_size, out_channels]，在这里被调整成 [batch_size, out_channels, 1, 1, 1]，以便与权重的形状匹配
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]
    # Execute by scaling the activations before and after the convolution.首先对输入 x 进行风格调制，然后执行卷积操作。
    if not fused_modconv:  # style经过调整后，与输入张量x相乘，调制输入的特征图，conv2d_resample是执行卷积的操作，它支持上采样up、下采样down和自定义滤波器resa_filter
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down,
                                            padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:  # 启用了去调制并且有噪声，则将噪声和去调制系数一起应用到卷积结果 x 上
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:  # 没有噪声，但启用了去调制，则仅应用去调制系数
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:  # 只有噪声，没有去调制，则只加上噪声
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings():  # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])  # 重塑输入为单批次的多通道输入
    w = w.reshape(-1, in_channels, kh, kw)  # 重塑权重；；使用批次维度作为分组groups=batch_size
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding,
                                        groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):  # 全连接层（FullyConnectedLayer），用于神经网络中的前向传播
    def __init__(self,
                 in_features,  # Number of input features
                 out_features,  # Number of output features.
                 bias=True,  # Apply additive bias before the activation function?是否在激活函数之前应用偏置
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.激活函数，默认是 'linear'线性（无激活）
                 lr_multiplier=1,  # Learning rate multiplier.学习率倍率，影响参数更新的速度，也会影响权重和偏置的初始化
                 bias_init=0,  # Initial value for the additive bias.偏置的初始值
                 ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn(
            [out_features, in_features]) / lr_multiplier)  # 权重矩阵，形状[out_fea, in_fea]，通过torch.randn随机初始化，并除lr来控制初始值的范围
        self.bias = torch.nn.Parameter(
            torch.full([out_features], np.float32(bias_init))) if bias else None  # 形状为[out_features]，根据 bias_init 的值初始化
        self.weight_gain = lr_multiplier / np.sqrt(in_features)  # 学习率倍率影响权重更新的尺度
        self.bias_gain = lr_multiplier  # 偏置的学习率倍率，通过 lr_multiplier 来控制

    def forward(self, x):
        w = self.weight.to(
            x.dtype) * self.weight_gain  # 权重self.weight被转换为输入数据x的数据类型（float32或float16），并乘weight_gain，确保权重更新时的尺度与学习率倍率相匹配
        b = self.bias
        if b is not None:  # 将偏置转换为与输入相同的数据类型
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain  # 需要对偏置进行缩放
        # 使用 torch.addmm，这是一种高效的矩阵乘法与加法操作，计算公式为 x = x @ w.T + b。w.t() 是权重矩阵的转置操作。b.unsqueeze(0) 将偏置扩展一个维度以便与矩阵加法匹配
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())  # 直接进行矩阵乘法 x @ w.T，不加偏置
            x = bias_act.bias_act(x, b,
                                  act=self.activation)  # 用bias_act.bias_act函数，将偏置b和激活函数应用于输出 x。激活函数的类型由self.activation决定
        return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 kernel_size,  # Width and height of the convolution kernel.
                 bias=True,  # Apply additive bias before the activation function?
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 up=1,  # Integer upsampling factor.
                 down=1,  # Integer downsampling factor.默认不下采样
                 resample_filter=[1, 3, 3, 1],
                 # Low-pass filter to apply when resampling activations.用于上/下采样的低通滤波器（默认为 [1, 3, 3, 1]
                 conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.是否将卷积结果裁剪到某个范围内
                 channels_last=False,  # Expect the input to have memory_format=channels_last?是否使用 channels_last 内存格式（
                 trainable=True,  # Update the weights of this layer during training?是否在训练期间更新权重（默认为 True，表示该层的权重可训练）
                 ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))  # 将低通滤波器注册为层的参数，它不是可训练的。
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(
            in_channels * (kernel_size ** 2))  # 用来调节卷积权重的缩放因子，1/sqrt(in_channels * kernel_size^2)，这是为了保持训练过程中数值的稳定性
        self.act_gain = bias_act.activation_funcs[
            activation].def_gain  # 激活函数的增益（默认的 gain），根据不同激活函数类型（如 relu、lrelu）来调整输出幅度

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:  # 初始化卷积核权重和偏置项。根据 trainable 标志决定这些参数是否是可学习的。如果不可训练，则将它们注册为 buffer（固定参数
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):  # x：输入张量，形状通常为 [batch_size, in_channels, height, width]；gain：可选的增益因子，默认为 1，用于控制输出的幅度
        w = self.weight * self.weight_gain  # w = self.weight * self.weight_gain：对卷积核的权重进行缩放，以保持数值的稳定性
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1)  # slightly faster设置为 True 时使用相关性代替卷积，以提高速度
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down,
                                            padding=self.padding, flip_weight=flip_weight)
        # 如果 up > 1，则在卷积之前对输入进行上采样；如果 down > 1，则在卷积之后对输出进行下采样
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


# gain 和 clamp：通过 act_gain 控制激活的增益，同时可以选择性地对输出进行裁剪（clamp
# 对卷积结果应用偏置和激活函数。如果提供了偏置 b，则在激活函数之前添加偏置。激活函数类型取决于 self.activation，如 ReLU、LeakyReLU 等
# ----------------------------------------------------------------------------

@persistence.persistent_class
# 在输入特征上进行卷积操作，并利用低频和高频信息来增强网络的表示能力
class FFCBlock(torch.nn.Module):
    def __init__(self,
                 dim,  # Number of output/input channels.dim 代表输入特征图和输出特征图的通道数相同
                 kernel_size,  # Width and height of the convolution kernel.卷积核的大小
                 padding,  # 卷积操作中使用的 padding 大小，用于控制输出特征图的尺寸
                 ratio_gin=0.75,  # 决定输入和输出的全局（高频）特征的比例
                 ratio_gout=0.75,
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 ):
        super().__init__()
        if activation == 'linear':  # 激活函数为 nn.Identity（即不使用激活
            self.activation = nn.Identity
        else:
            self.activation = nn.ReLU
        self.padding = padding
        self.kernel_size = kernel_size
        self.ffc_block = FFCResnetBlock(dim=dim,  # 用于对输入特征图进行频率分解卷积操作。FFCResnetBlock 在底层执行了两次带有残差连接的卷积操作
                                        padding_type='reflect',  # 表示卷积时使用反射填充
                                        norm_layer=nn.SyncBatchNorm,  # 指定使用同步批量归一化层
                                        activation_layer=self.activation,  # 在卷积操作后使用的激活函数
                                        dilation=1,
                                        ratio_gin=ratio_gin,  # 控制输入和输出的低频与高频通道的比例。
                                        ratio_gout=ratio_gout)

        self.concat_layer = ConcatTupleLayer()  # 将 FFC 模块生成的低频和高频特征图拼接在一起

    def forward(self, gen_ft, mask, fname=None):
        x = gen_ft.float()  # 输入的特征图，形状为 [batch_size, channels, height, width]，将输入 gen_ft 转换为浮点型数据
        # 计算全局（高频）通道数量，并将 gen_ft 分解为局部（低频）特征图 x_l 和全局（高频）特征图 x_g；x_g 是最后 global_in_num 个通道，表示全局特征
        x_l, x_g = x[:, :-self.ffc_block.conv1.ffc.global_in_num], x[:, -self.ffc_block.conv1.ffc.global_in_num:]
        id_l, id_g = x_l, x_g  # x_l代表前 dim - global_in_num 个通道，表示局部特征；
        # 将分解出的局部和全局特征输入ffc_block，进行两次频率分解卷积操作，在每次卷积操作后，将输出的局部特征与输入的局部特征相加，输出的全局特征与输入的全局特征相加（残差连接
        x_l, x_g = self.ffc_block((x_l, x_g), fname=fname)
        x_l, x_g = id_l + x_l, id_g + x_g
        x = self.concat_layer((x_l, x_g))  # 使用 self.concat_layer 将卷积得到的局部特征 x_l 和全局特征 x_g 重新拼接在一起，形成新的特征图
        # 将拼接后的输出特征与最初的输入特征 gen_ft 相加（残差连接），从而得到最终的输出
        return x + gen_ft.float()


# ----------------------------------------------------------------------------

@persistence.persistent_class
# 用于编码网络末端的模块。它可能还会应用条件映射。用于将输入特征图通过卷积、全连接、条件映射等操作，生成潜在向量 z。它包含条件处理、MiniBatch 标准差、卷积等功能
class EncoderEpilogue(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 cmap_dim,  # Dimensionality of mapped conditioning label, 条件映射向量的维度，0 = no label.用于条件生成或条件编码
                 z_dim,  # Output Latent (Z) dimensionality.最终输出的潜在向量的大小
                 resolution,  # Resolution of this block.分辨率
                 img_channels,  # Number of input color channels.
                 architecture='resnet',  # Architecture: 'orig原始架构', 'skip', 'resnet'.
                 mbstd_group_size=4,
                 # Group size for the minibatch standard deviation layer, None = entire minibatch（mini-batch）.MiniBatch标准差层的组大小
                 mbstd_num_channels=1,
                 # Number of features for the minibatch standard deviation layer, 0 = disable.MiniBatch 标准差层的通道数
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 conv_clamp=None,
                 # Clamp the output of convolution layers to +-X, None = disable clamping.卷积层输出的夹紧值, None=禁用夹紧
                 ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(self.img_channels, in_channels, kernel_size=1, activation=activation)

        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size,
                                       num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        # 核心卷积层。输入通道数为 in_channels + mbstd_num_channels（因为可能加入 MiniBatch 标准差通道），然后再输出相同的 in_channels
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation,
                                conv_clamp=conv_clamp)
        # 全连接层，用于将卷积特征转化为潜在空间的向量 z，并应用了 Dropout，避免过拟合
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), z_dim, activation=activation)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, cmap, force_fp32=False):  # x: 输入特征图，形状为[batch_size, in_chann, reso, reso]，条件映射向量（用于条件生成），形状为 [batch_size, cmap_dim]
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])  # [NCHW]
        _ = force_fp32  # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.将输入 x 转换为指定的数据类型和内存格式，
        x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.如果启用了 MiniBatch 标准差层，计算并加入标准差特征
        if self.mbstd is not None:
            x = self.mbstd(x)
        const_e = self.conv(x)  # 通过卷积层提取特征 const_e，然后将其展平并输入全连接层生成潜在向量 z，之后应用 Dropout
        x = self.fc(const_e.flatten(1))
        x = self.dropout(x)

        # Conditioning.将潜在向量 x 与条件向量 cmap 进行逐元素相乘，随后沿 dim=1 维度求和，并做归一化。
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x, const_e  # x: 最终的潜在向量（可能经过条件映射）。const_e: 通过卷积层提取的特征图


# ----------------------------------------------------------------------------

from training.Gsop import GSoP
@persistence.persistent_class
class EncoderBlock(torch.nn.Module):  # 用于处理输入图像或特征图并生成特征表示
    def __init__(self,
                 in_channels,  # Number of input channels, 0 = first block.
                 tmp_channels,  # Number of intermediate channels. 中间特征的通道数.
                 out_channels,  # Number of output channels.
                 resolution,  # Resolution of this block.
                 img_channels,  # Number of input color channels.输入图像的颜色通道数
                 first_layer_idx,  # Index of the first layer.
                 architecture='skip',  # Architecture: 'orig', 'skip', 'resnet'.
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.低通滤波器用于上/下采样
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.卷积层输出值的夹紧
                 use_fp16=False,  # Use FP16 for this block?是否使用 FP16 计算精度
                 fp16_channels_last=False,
                 # Use channels-last memory format with FP16?是否在 FP16 下使用 `channels_last` 内存格式
                 freeze_layers=0,  # Freeze-D: Number of layers to freeze.冻结前多少层（即这些层不参与训练）
                 num_heads=8,  # 2++++新增参数：注意力头数
                 bias=True  # 2+++++新增参数：注意力层中的 bias 设置
                 ):
        # 断言语句，用于在代码执行时检查某些条件是否满足
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()  # 调用了父类 torch.nn.Module 的构造函数。
        # 设置类属性
        self.in_channels = in_channels
        self.resolution = resolution  # 它指的是特征图的高度和宽度。
        self.img_channels = img_channels + 1
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture  # 保存第一个层的索引。该索引可以用于确定当前层在整个网络中的位置。
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        # register_buffer 注册了一个称为 resample_filter 的常量（即不会参与模型的参数更新
        self.num_layers = 0  # 用于记录当前 block 中的层数

        def trainable_gen():  # 生成器函数，用于确定每一层是否是可训练的
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()

        if in_channels == 0:  # 如果输入是图像而不是特征图，定义一个 1x1 的卷积层用于从 RGB 图像中提取特征，并将 img_channels 转换为 tmp_channels
            self.fromrgb = Conv2dLayer(self.img_channels, tmp_channels, kernel_size=1, activation=activation,
                                       trainable=next(trainable_iter), conv_clamp=conv_clamp,
                                       channels_last=self.channels_last)
        # conv0 和 conv1 是核心的卷积操作，conv1 在卷积后进行下采样（down=2），即将分辨率减少一半
        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
                                 trainable=next(trainable_iter), conv_clamp=conv_clamp,
                                 channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
                                 trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp,
                                 channels_last=self.channels_last)

        if architecture == 'resnet':  # 0
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                                    trainable=next(trainable_iter), resample_filter=resample_filter,
                                    channels_last=self.channels_last)
        # self.gsop = GSoP.GSoP(out_channels, attention='+', att_dim=128)   # 4++++ gsop

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:  # 检查输入的形状是否符合预期，确保其是 [batch_size, in_channels, resolution, resolution] 的格式
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0:  # 这是网络的第一个 block（即 in_channels == 0），则从 RGB 图像中提取特征
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None
        # 如果架构是 skip，对图像进行下采样以保持分辨率一致性
        # Main layers.
        if self.architecture == 'resnet':  # 0
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)  # 之后，经过 conv0 进行卷积并保存中间特征 feat，然后通过 conv1 进行下采样

            feat = x.clone()
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)  # 将跳跃连接的结果 y 和 conv1 的输出 x 相加，得到最终的特征
        else:  # 简单地通过 conv0 和 conv1 进行卷积和下采样，并且保存中间特征 feat。  #  111
            x = self.conv0(x)
            feat = x.clone()
            x = self.conv1(x)

            # x = self.gsop(x)  # 4++++

        # x = x.to(dtype)  # 4++++
        assert x.dtype == dtype

        return x, img, feat


# x：当前 block 的输出特征图（可能是下一个 block 的输入；img：下采样后的图像（在 skip 架构中；feat：当前 block 中的中间特征，用于在进一步的层中进行计算
# ----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 w_dim,  # Intermediate latent (W) dimensionality.用于风格化操作的潜在向量 w 的维度
                 resolution,  # Resolution of this layer.决定输出图像或特征图的大小
                 kernel_size=3,  # Convolution kernel size.
                 up=1,  # Integer upsampling factor.1，表示没有上采样
                 use_noise=True,  # Enable noise input?在生成网络中可以引入随机性
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,
                 # Clamp the output of convolution layers to +-X, None = disable clamping.用于控制卷积层输出的最大值，避免数值过大。
                 channels_last=False,  # Use channels_last format for the weights?
                 ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)  # 全连接层，负责将风格向量 w 投影为输入通道的缩放因子（用于调制卷积核
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(
            memory_format=memory_format))  # 卷积核权重，随机初始化
        if use_noise:  # 启用了噪声注入，noise_const 是用于生成固定噪声的常量，而 noise_strength 控制噪声的强度
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))  # 卷积操作后的偏置量

    # x: 输入特征图，形状为 [batch_size, in_channels, height, width]；风格向量 w，用于调节卷积核权重；控制是否添加噪声及如何添加。可选值包括 'random'（随机噪声）、'const'（固定噪声）和 'none'（无噪声）
    # fused_modconv: 是否将调制卷积操作与卷积操作融合，提升计算效率；gain: 控制激活输出的缩放因子
    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])  # 确保输入的特征图形状匹配期望的卷积输入形状
        styles = self.affine(w)  # 使用全连接层 affine 将风格向量 𝑤转换为卷积核的调制因子 styles。这些因子用于调节卷积核权重

        noise = None  # 生成随机噪声，并按 noise_strength 的比例缩放。如果是 'const'，则使用预定义的噪声常量 noise_const
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution],
                                device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1)  # slightly faster
        # 调用 modulated_conv2d 函数执行调制卷积x 是输入特征图。weight 是卷积核的权重，由风格因子 styles 调制。noise 是噪声（如果启用）。up 是上采样因子（决定是否上采样）。
        # padding 用于保持卷积后输出的尺寸。resample_filter 用于对卷积结果进行低通滤波，确保平滑。flip_weight 是卷积核的翻转标志，用于加速计算
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
                             padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight,
                             fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None  # 如果有 conv_clamp，则通过 clamp 将输出限制在特定范围内，避免数值溢出
        x = F.leaky_relu(x, negative_slope=0.2, inplace=False)  # 使用 Leaky ReLU 作为激活函数（默认情况下，negative_slope=0.2）
        if act_gain != 1:
            x = x * act_gain  # act_gain 控制激活后的输出缩放
        if act_clamp is not None:
            x = x.clamp(-act_clamp, act_clamp)
        return x


# 该层在图像生成任务中能够将潜在空间中的风格向量 𝑤通过调制卷积映射为图像特征。它的设计允许生成具有不同风格和细节的高分辨率图像。
# ----------------------------------------------------------------------------

@persistence.persistent_class
# 通过 FFCBlock 执行一个跳跃连接层（skip connection）。它被设计为在输入和输出的特征图之间执行特定的运算，使特征图保留一些信息，同时在中间通过 FFCBlock 进行处理。
# 输入特征图中的信息不会被完全抛弃，而是通过跳跃连接的方式保留，同时引入了新的经过处理的特征
class FFCSkipLayer(torch.nn.Module):
    def __init__(self,
                 dim,  # Number of input/output channels.通常特征图的深度（即通道数）是输入和输出相同的
                 kernel_size=3,  # Convolution kernel size.决定卷积操作的感受野大小
                 ratio_gin=0.75,  # 分别控制全局和局部通道的比率。这些参数来自于 FFCBlock，该模块将输入特征分为全局和局部特征图，分别处理
                 ratio_gout=0.75,
                 ):
        super().__init__()
        self.padding = kernel_size // 2  # 计算出卷积层需要的填充大小。这确保输入和输出的空间维度保持一致。
        # 实例化了一个 FFCBlock。FFCBlock 是一种专门用于处理局部和全局特征的模块。它将输入的特征图分为局部和全局两部分，分别进行处理，再合并。padding: 保证输入输出的尺寸一致
        self.ffc_act = FFCBlock(dim=dim, kernel_size=kernel_size, activation=nn.ReLU,
                                padding=self.padding, ratio_gin=ratio_gin, ratio_gout=ratio_gout)

    # gen_ft: 输入的特征图，形状为 [batch_size, dim, height, width]，表示输入的特征张量；掩码，用于局部特征和全局特征的处理；文件名参数，通常用于调试或记录过程中传递的附加信息
    def forward(self, gen_ft, mask, fname=None):
        # 通过 FFCBlock 对输入特征图 gen_ft 进行处理。FFCBlock 内部会将输入特征拆分为局部和全局部分，分别进行卷积、激活等操作，然后再合并这些特征
        # mask 也会影响特征图的处理，帮助控制哪些区域应该用局部卷积，哪些应该用全局卷积。
        x = self.ffc_act(gen_ft, mask, fname=fname)
        return x  # 返回处理过的特征图 x，相当于执行了一个 "skip" 操作，即在原始输入特征上叠加了一些经过处理的额外信息。


# ----------------------------------------------------------------------------

@persistence.persistent_class
# 将特征映射转换为 RGB 图像，常用于生成网络的最后阶段。它将高维的特征图转换成实际的图像，通过卷积和特征调制的方式，输出期望的颜色通道（通常是RGB，输出通道数为3
class ToRGBLayer(torch.nn.Module):
    # w_dim: 输入的潜在向量 w 的维度。通常这是一个中间表示，源自更早的潜在向量 z 经过多层变换得到的 w
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels,
                                          bias_init=1)  # 将潜在向量w转换为与输入特征图通道数匹配的样式向量style。w经过affine层后生成一组调制参数，来控制卷积权重。
        memory_format = torch.channels_last if channels_last else torch.contiguous_format  # 确定权重的存储格式。如果channels_last则使用channels_las格式来加速计算。否则用contiguous格式
        # 卷积的权重矩阵，大小为 [out_channels, in_channels, kernel_size, kernel_size]。通过随机初始化，随后会通过 w 向量调制这些权重
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))  # 输出特征的偏置，初始化为 0。用于调整卷积的结果。
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))  # 权重增益因子。通过缩放权重，避免因输入通道数较大时卷积结果过大，稳定训练过程

    def forward(self, x, w, fused_modconv=True):
        # x: 输入的特征图，形状为 [batch_size, in_channels, height, width]；w: 潜在向量 w，用于生成样式向量。fused_modconv: 如果为 True，使用融合的调制卷积进行高效计算。否则会分步执行调制和卷积
        styles = self.affine(
            w) * self.weight_gain  # 通过affine层将输入的潜在向量w转换为调制向量styles。调制向量大小为[batch_size, in_channels]，乘以wei_ga，以控制权重的幅度，防止数值过大导致梯度爆炸
        # 卷积操作，卷积核权重会被 styles 向量调制（即逐个元素地乘以样式向量）。demodulate=False: 这里不执行反调制操作；used_modconv: 如果为 True，则会将调制和卷积操作融合为一个更高效的操作
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        # bias_act 函数会对卷积的结果 x 添加偏置，并应用激活函数。在这个过程中，还可以选择性地对输出进行裁剪（由 conv_clamp 控制；self.bias.to(x.dtype)：将偏置转换为与输入特征图相同的数据类型。
        # clamp=self.conv_clamp：如果 conv_clamp 不为 None，则对输出进行裁剪，确保其在指定范围内。
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x


# ----------------------------------------------------------------------------
@persistence.persistent_class
# 用于生成特定分辨率的图像，并结合多种架构来控制数据流的方式。通过一系列卷积、跳跃连接（skip connections）和特征映射（RGB 输出）来处理输入特征图，并输出合成后的特征图

class SynthesisBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels, 0 = first block.输入为常数特征图
                 out_channels,  # Number of output channels.决定了经过这个块后输出特征图的通道数
                 w_dim,  # Intermediate latent (W) dimensionality.用于控制特征图的样式
                 resolution,  # Resolution of this block.
                 img_channels,  # Number of output color图像 channels.
                 is_last,  # Is this the last block?
                 architecture='skip',  # Architecture: 'orig', 'skip', 'resnet'.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.用于处理卷积后的特征图
                 conv_clamp=None,
                 # Clamp the output of convolution layers to +-X, None = disable clamping.将卷积输出限制在 +-X 的范围内，避免数值溢出
                 use_fp16=False,  # Use FP16 for this block?
                 fp16_channels_last=False,  # Use channels-last memory format with FP16?决定是否使用 channels_last 的内存格式。
                 **layer_kwargs,  # Arguments for SynthesisLayer.
                 ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0
        self.res_ffc = {4: 0, 8: 0, 16: 0, 32: 1, 64: 1, 128: 1, 256: 1, 512: 1}

        if in_channels != 0 and resolution >= 8:
            self.ffc_skip = nn.ModuleList()  # 创建一组 FFCSkipLayer 层，用于在不同分辨率下添加全局跳跃连接。res_ffc 是一个字典，用于确定不同分辨率下跳跃连接的数量
            for _ in range(self.res_ffc[resolution]):
                # print(resolution)
                self.ffc_skip.append(FFCSkipLayer(dim=out_channels))

        if in_channels == 0:  # 用常数张量作为输入
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            # 第一个卷积层。它进行上采样（up=2）操作，使特征图从较低分辨率提升到较高分辨率；conv0 是一个 SynthesisLayer
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim * 3, resolution=resolution, up=2,
                                        resample_filter=resample_filter, conv_clamp=conv_clamp,
                                        channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1
        # 第二个卷积层，它继续处理输出特征图
        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim * 3, resolution=resolution,
                                    conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':  # 生成 RGB 输出层。这一层将特征图转化为 RGB 图像。
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim * 3,
                                    conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':  # skip 用于创建残差连接，它通过一个 1x1 卷积对输入进行处理，并在分辨率上采样后与输出相加
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                                    resample_filter=resample_filter, channels_last=self.channels_last)

        self.eca = ECALayer()  # 1+++加入 ECA 模块
        self.depthwise_separable = DepthwiseSeparableConv(out_channels, out_channels, alpha=0.5)  # 7+++深度可分离卷积


    def forward(self, x, mask, feats, img, ws, fname=None, force_fp32=False, fused_modconv=None, **layer_kwargs):
        # x输入特征图。如果这是第一个块，则x是恒定输入；否则为前一层传递的特征图；feats: 前一层生成的特征图，在某些情况下用于跳跃连接；ws: 潜在变量W的列表，用于调制卷积操作中的特征图
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32  #x:8,512,4,4,folat32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format  # 决定张量的内存布局
        if fused_modconv is None:
            with misc.suppress_tracer_warnings():  # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        x = x.to(dtype=dtype, memory_format=memory_format)  # 8 512 4 4
        x_skip = feats[self.resolution].clone().to(dtype=dtype, memory_format=memory_format)  # 8 512 8 8
        # 从 feats 中获取与当前分辨率匹配的特征图，并将其复制为 x_skip，后面可能在跳跃连接或 FFC 操作中使用
        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, ws[1], fused_modconv=fused_modconv, **layer_kwargs)

            x = self.eca(x.to(dtype=dtype))  #1+++ 在第一个卷积后应用 ECA0
            x = self.depthwise_separable(x)

        elif self.architecture == 'resnet':  # 通过 skip 跳跃连接生成残差分支 y
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, ws[0].clone(), fused_modconv=fused_modconv, **layer_kwargs)  # 执行 conv0 卷积层，将 ws[0] 用于特征调制

            # x = self.eca(x.to(dtype=dtype))  # 1+++ 在 conv0 后应用 ECA0
            # x = self.depthwise_separable(x)  # 7+++ 加入深度可分离卷积


            if len(self.ffc_skip) > 0:  # 如果存在 ffc_skip 层，插值 mask 以匹配 x_skip 的大小，将 x 和 x_skip 相加，然后通过 ffc_skip 层处理结果
                mask = F.interpolate(mask, size=x_skip.shape[2:], )
                z = x + x_skip
                for fres in self.ffc_skip:
                    z = fres(z, mask)
                x = x + z
            else:  # 执行 conv1，并将 x 与残差连接 y 进行加和
                x = x + x_skip
            x = self.conv1(x, ws[1].clone(), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)

            # x = self.eca(x.to(dtype=dtype))  # 在  1+++conv1 后应用 ECA0
            # x = self.depthwise_separable(x)  # 7+++ 加入深度可分离卷积0

            x = y.add_(x)
        else:  # 执行 conv0 并将结果与 x_skip 相加（如果存在 ffc_skip，则进行相应的操作1

            x = self.conv0(x, ws[0].clone(), fused_modconv=fused_modconv, **layer_kwargs)  # 8 512 8 8

            x = self.eca(x.to(dtype=dtype))  # 1+++ 在 conv0 后应用 ECA1
            x = self.depthwise_separable(x)  # 7+++ 加入深度可分离卷积    可以做消融

            if len(self.ffc_skip) > 0:  # 32 64 128 256
                mask = F.interpolate(mask, size=x_skip.shape[2:], )
                z = x + x_skip

                for fres in self.ffc_skip:
                    z = fres(z, mask)
                x = x + z
            else:
                x = x + x_skip

            x = self.conv1(x, ws[1].clone(), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.eca(x.to(dtype=dtype))  # 在 1+++ conv1 后应用 ECA1
            x = self.depthwise_separable(x)  # 7+++ 加入深度可分离卷积

        # ToRGB.
        if img is not None:     # 111
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':  # 111
            y = self.torgb(x, ws[2].clone(), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y  # 在生成的RGB图像y中,如果已经存在图像，则将其与img加和；否则用y为img

        x = x.to(dtype=dtype)
        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img  # 返回特征图 x 和最终的图像 img。x 使用原始数据类型（可能为 FP16），而 img 总是以 torch.float32 类型返回

# ----------------------------------------------------------------------------

@persistence.persistent_class
# 从潜在向量 Z 开始，将其映射为特征图，使用卷积层进行特征提取，并通过风格调制控制生成图像的风格。它通过跳跃连接和多层卷积逐渐生成特征图，最终在 skip 架构中生成 RGB 图像
# 用于在生成器的起始阶段生成图像特征和最终图像。
class SynthesisForeword(torch.nn.Module):
    def __init__(self,
                 z_dim,  # Output Latent (Z) dimensionality.潜在向量 Z 的维度，定义了生成图像的特征
                 resolution,  # Resolution of this block.
                 in_channels,  # 输入特征图的通道数。
                 img_channels,  # Number of input color channels.
                 architecture='skip',  # Architecture: 'orig', 'skip', 'resnet'.
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.用于卷积层的激活

                 ):
        super().__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture
        # 处理z_dim的潜在向量，将其从[batch_size, z_dim]转换为一个大小[batch_size, (z_dim // 2) * 4 * 4]的向量;将潜在向量扩展为小分辨率（4x4）的特征图，同时应用激活函数
        self.fc = FullyConnectedLayer(self.z_dim, (self.z_dim // 2) * 4 * 4, activation=activation)
        # 用于将扩展的特征图进一步卷积处理，提升图像的质量,该卷积层通过将特征图从输入通道卷积到输出通道，同时使用风格调制（基于 ws 向量）来控制卷积的行为
        self.conv = SynthesisLayer(self.in_channels, self.in_channels, w_dim=(z_dim // 2) * 3, resolution=4)

        if architecture == 'skip':  # 启用 ToRGBLayer，将卷积层输出的特征图转换为 RGB 图像
            self.torgb = ToRGBLayer(self.in_channels, self.img_channels, kernel_size=1, w_dim=(z_dim // 2) * 3)

    def forward(self, x, ws, feats, img, force_fp32=False):
        misc.assert_shape(x, [None, self.z_dim])  # [NC]检查输入x的形状是否为[batch_size, z_dim]。
        _ = force_fp32  # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        x_global = x.clone()  # 将潜在向量x复制为x_global，以便用于风格调制
        # ToRGB.
        x = self.fc(x)  # 使用全连接层 fc 将潜在向量 x 映射到大小为 z_dim // 2 通道、4x4 分辨率的特征图
        x = x.view(-1, self.z_dim // 2, 4, 4)  # 使用 view 将输出的 1D 张量转换为形状为 [batch_size, z_dim // 2, 4, 4] 的 4D 张量
        x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        x_skip = feats[4].clone()  # 从 feats 中获取与分辨率 4x4 对应的跳跃连接特征图 x_skip
        x = x + x_skip  # 将 x_skip 和 x 进行相加，这是一种跳跃连接机制，允许不同层次的特征图相互融合。

        mod_vector = []  # mod_vector 用于风格调制。mod_vector 是由 ws 中的特定潜在变量和 x_global 组成
        mod_vector.append(ws[:, 0])  # 第一个 mod_vector 是 ws[:, 0] 和 x_global 的拼接
        mod_vector.append(x_global.clone())
        mod_vector = torch.cat(mod_vector, dim=1)

        x = self.conv(x, mod_vector)  # 用 mod_vector 调制 conv 卷积层的权重，对特征图进行卷积处理

        mod_vector = []
        mod_vector.append(ws[:, 2 * 2 - 3])  # 第二个 mod_vector 是 ws[:, 2*2-3] 和 x_global 的拼接
        mod_vector.append(x_global.clone())
        mod_vector = torch.cat(mod_vector, dim=1)

        if self.architecture == 'skip':  # 调用 torgb 将特征图转换为 RGB 图像 img
            img = self.torgb(x, mod_vector)
            img = img.to(dtype=torch.float32, memory_format=torch.contiguous_format)

        assert x.dtype == dtype
        return x, img


# ----------------------------------------------------------------------------

@persistence.persistent_class
# orig（传统架构）:输入 -> conv0 -> conv1 -> 输出
# skip（跳跃连接架构）:输入 -> fromrgb -> conv0 -> conv1 -> 跳跃到输出
# resnet（残差架构）:输入 -> skip -> conv0 -> conv1 -> 残差相加 -> 输出
# 从图像特征中提取信息，并判断输入的图像是否是真实图像或生成的图像。该模块支持不同架构 ('orig', 'skip', 'resnet')，并通过卷积操作逐步下采样图像。
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels, 0 = first block.
                 tmp_channels,  # Number of intermediate channels.中间卷积层的通道数。通常用于提取和转换特征
                 out_channels,  # Number of output channels.
                 resolution,  # Resolution of this block.
                 img_channels,  # Number of input color channels.
                 first_layer_idx,  # Index of the first layer. 当前层的索引，用于在冻结层数（freeze_layers）的判断过程中，确定哪些层是可训练的。
                 architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.用于减少信息丢失
                 conv_clamp=None,
                 # Clamp the output of convolution layers to +-X, None = disable clamping.用于将卷积输出限制在指定的范围内，防止数值溢出。如果为 None，则不进行限制。
                 use_fp16=False,  # Use FP16 for this block?
                 fp16_channels_last=False,  # Use channels-last内存布局（优化内存访问效率 memory format with FP16?
                 freeze_layers=0,  # Freeze-D: Number of layers to freeze.冻结前 n 层的参数，不进行训练，可以用来提高模型的稳定性
                 ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels + 1
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0

        def trainable_gen():  # 一个生成器函数，用于决定每一层是否是可训练的。当某一层的索引大于等于 freeze_layers 时，该层是可训练的；否则，层的参数将被冻结
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            # 处理输入图像的第一层卷积。fromrgb 层将把输入的 RGB 图像（或包含 mask 的图像）通过 1x1 卷积转换为具有tmp_channels 数量的中间特征图。
            self.fromrgb = Conv2dLayer(self.img_channels, tmp_channels, kernel_size=1, activation=activation,
                                       trainable=next(trainable_iter), conv_clamp=conv_clamp,
                                       channels_last=self.channels_last)
        # 一个标准的 3x3 卷积层，输入为 tmp_channels 通道，输出也是 tmp_channels 通道。该层在经过卷积后会应用激活函数（例如 lrelu
        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
                                 trainable=next(trainable_iter), conv_clamp=conv_clamp,
                                 channels_last=self.channels_last)
        # 第二个 3x3 卷积层，与conv0类似，这层同时会下采样输入特征图（通过down=2），将分辨率降低一半，并将输出通道转换为out_channels。这一步通过卷积和下采样来提取高层次的特征
        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
                                 trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp,
                                 channels_last=self.channels_last)
        # 一个 1x1 的卷积层，用于实现残差连接，将输入直接传递到下一个模块。该操作的目的在于通过 down=2 进行下采样，并与主路径的卷积结果相加
        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                                    trainable=next(trainable_iter), resample_filter=resample_filter,
                                    channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):  # 负责在判别器中对输入特征图（x）和图像（img）进行处理，并将其传递给卷积层以提取特征
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.确保其形状为 [batch_size, in_channels, resolution, resolution]，然后将输入特征图 x 转换为相应的数据类型 (dtype) 和内存格式 (memory_format
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.fromRGB层将RGB图像输入转换为特征图，用于第一个卷积块的输入
        if self.in_channels == 0 or self.architecture == 'skip':  # 图像输入会经过 fromRGB 层
            misc.assert_shape(img, [None, self.img_channels, self.resolution,
                                    self.resolution])  # 检查img形状，确保其维度为[batch_size, img_channels, resolution, resolution]
            img = img.to(dtype=dtype, memory_format=memory_format)  # 将图像转换为 dtype 和 memory_format
            y = self.fromrgb(img)  # 通过 fromrgb 层提取图像特征，并将提取的特征与输入特征 x 相加（如果 x 不为 None）
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img,
                                         self.resample_filter) if self.architecture == 'skip' else None  # 还会将图像通过down层下采样，用于后续跳跃连接

        # Main layers.
        if self.architecture == 'resnet':  # 额外添加残差路径（跳跃连接
            y = self.skip(x, gain=np.sqrt(0.5))  # 通过 skip 层对输入特征进行 1x1 卷积和下采样操作，将其存储在变量 y 中。
            x = self.conv0(x)  # 对 x 通过 conv0 和 conv1 两个卷积层处理，每次都会进行卷积和激活，并使用 gain=np.sqrt(0.5) 来平衡输出
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)  # 将残差路径 y 和主路径的结果 x 相加，实现残差连接。
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img


# ----------------------------------------------------------------------------

@persistence.persistent_class
# 帮助模型区分真实图像和生成图像。通过引入小批量标准差（minibatch standard deviation特征来增强模型对小批量样本间的统计差异的敏感性，帮助模型更好地检测生成图像中的不一致性
class MinibatchStdLayer(torch.nn.Module):
    # group_size：将小批量中的样本分为若干组，每组计算一次标准差;num_channels：每个组中额外生成的特征图数量，默认为 1，通常用于表示一组的标准差
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape  # N 表示 batch size（小批量大小），C 是输入特征图的通道数，H 和 W 分别是特征图的高度和宽度
        with misc.suppress_tracer_warnings():  # as_tensor results are registered as constants确保 G 不会超过批量大小 N，并且根据 self.group_size 参数决定每组包含的样本数
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels  # 将输入 x 进行重塑，将 batch 中的样本分成 G 组，并且将输入通道数 C 划分成 F 组，每组包含c = C // F 个通道。目的是为每一组计算标准差
        c = C // F
        # y的形状最初是 [G, F, 1, 1]，其中 G 是 batch 大小，F 是标准差的通道数。使用 y.repeat(G, 1, H, W) 将 y 的形状扩展为 [G, F, H, W]
        # 使得 y 在高度（H）和宽度（W）上与输入特征图 x 保持一致。在通道维度 dim=1 上拼接，形成形状为 [N, C+F, H, W] 的新特征图。
        y = x.reshape(G, -1, F, c, H,
                      W)  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.计算每组通道的均值，从每个样本中减去该均值，使其中心化。计算组内的统计差异
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.先计算方差，再取平方根得到标准差。1e-8 是为了避免数值不稳定性，即防止除以0
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.将标准差在每个通道和像素维度上进行平均，得到[G, F]的标准差
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions. 将其变形为 [G, F, 1, 1]，以便稍后与原始特征图拼接。
        y = y.repeat(G, 1, H,
                     W)  # [NFHW]   Replicate over group and pixels.将y中的标准差特征图重复，确保其形状与原始输入特征图x匹配。y的形状为 [N, F, H, W]可以与输入x进行通道维度上的拼接
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
# 将特征图转化为最终的判别输出
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 cmap_dim,  # Dimensionality of mapped conditioning label, 0 = no label.条件标签的映射维度，用于条件生成任务
                 resolution,  # Resolution of this block.
                 img_channels,  # Number of input color channels.
                 architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
                 mbstd_group_size=4,
                 # Group size for the minibatch standard deviation layer, None = entire minibatch.用于 MinibatchStdLayer 的组大小，用于统计标准差
                 mbstd_num_channels=1,
                 # Number of features for the minibatch standard deviation layer, 0 = disable.MinibatchStdLayer 生成的特征通道数
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 conv_clamp=None,
                 # Clamp the output of convolution layers to +-X, None = disable clamping.卷积层输出的截断范围，如果设置为 None，则不进行截断
                 ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':  # 额外添加一个 fromrgb 层，用于将 RGB 图像直接转化为特征图。
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        # 添加一个 MinibatchStdLayer，用于计算每个小批量数据的标准差，并将其作为额外的统计特征添加到输入中。
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size,
                                       num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        # 将输入通道（包括标准差通道）转化为输出特征图，卷积核大小为 3x3。
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation,
                                conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels,
                                      activation=activation)  # 将卷积层输出的特征展平并转化为固定长度的向量
        self.out = FullyConnectedLayer(in_channels,
                                       1 if cmap_dim == 0 else cmap_dim)  # 根据是否存在cmap_dim，输出一个标量（表示对抗损失），或条件标签对应的向量。

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])  # [NCHW]
        _ = force_fp32  # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.将输入图像 img 转换为特征图，并与输入特征图 x 相加。img 是原始输入图像，x 是来自前一个判别器块的特征图。fromrgb 层使用一个 1x1 的卷积将图像转化为特征图，便于与其他特征融合
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)  # 计算小批量标准差并添加到特征图中
        x = self.conv(x)  # 通过一个3x3卷积操作对输入进行处理，得到新的特征图。卷积层的输入是加入了标准差特征图的特征图，输出具有与输入相同的空间维度（但通道数可能会不同
        x = self.fc(x.flatten(1))  # 将卷积层的输出展平为一维向量，输入全连接层，将其转化为固定长度的特征向量
        x = self.out(x)  # 输出层根据是否有条件标签来确定输出形式.cmap_dim > 0,将判别器输出与条件标签进行点积（通过 cmap 进行条件标签的乘积），得到条件化的判别输出
        # cmap_dim = 0：输出一个标量，表示判别器的输出值（通常是图像是否真实的概率）。
        # Conditioning.cmap 表示映射后的条件向量，其维度为 [N, cmap_dim]。该条件标签通过点积与特征向量进行融合，然后输出一个标量作为最终结果
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

# ----------------------------------------------------------------------------
