# Modified from https://github.com/NVlabs/stylegan2-ada-pytorch

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from training.ECA import ECALayer
from training.models import *


# ----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):  #将z和c转换为w
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality, 0 = no latent.随机潜在向量z
                 c_dim,  # Conditioning label (C) dimensionality, 0 = no label.条件向量c
                 w_dim,  # Intermediate latent (W) dimensionality.中间表示w的维度
                 num_ws,  # Number of intermediate latents to output, None = do not broadcast.输出的中间表示数量
                 num_layers=8,  # Number of mapping layers.网络层数，通常映射网络有多个全连接层
                 embed_features=None,  # Label embedding dimensionality, None = same as w_dim.
                 layer_features=None,  # Number of intermediate features in the mapping layers, None = same as w_dim.
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
                 w_avg_beta=0.995,  # Decay for tracking the moving average of W during training, None = do not track.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta  #训练过程中用于追踪w的移动平均值，影响生成图像的质量

        if embed_features is None:  #类别标签（conditioning label）嵌入后的特征维度
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:  #用于控制每一层的特征数量
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):  #构建了映射网络中的每一层全连接层
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation,
                                        lr_multiplier=lr_multiplier)  #使用指定的激活函数（如 'lrelu'）和学习率缩放因子 lr_multiplier。
            setattr(self, f'fc{idx}', layer)  #将生成的每一层赋给对象的属性 fc{idx}，确保网络的层结构按顺序保存

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):  #输入z 和c 被规范化后通过多层全连接层生成w
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.移动平均w 通过参数 w_avg_beta 进行更新
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x


# ----------------------------------------------------------------------------
'''
class SelfAttention(nn.Module):  # 5++++
    def __init__(self, in_dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)  # 归一化层
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape  # (batch, channels, height, width)
        x = x.view(B, C, -1).permute(0, 2, 1)  # 变换为 (batch, seq_len, channels)
        x = x.to(dtype=self.norm.weight.dtype)
        x = self.norm(x)
        x, _ = self.attn(x, x, x)  # Self-Attention
        x = x.permute(0, 2, 1).view(B, C, H, W)  # 变回 (batch, channels, height, width)
        return x
'''
@persistence.persistent_class
class EncoderNetwork(torch.nn.Module):  #从输入图像中提取特征，并将这些特征映射为潜在空间的表示
    def __init__(self,
                 c_dim,  # Conditioning label (C) dimensionality.
                 z_dim,  # Input latent (Z) dimensionality.
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 architecture='orig',  # Architecture: 'orig', 'skip', 'resnet'.
                 channel_base=16384,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=0,  # Use FP16 for the N highest resolutions.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.条件映射维度
                 block_kwargs={},  # Arguments for DiscriminatorBlock.
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 epilogue_kwargs={},  # Arguments for EncoderEpilogue.
                 ):
        super().__init__()
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))  ## 分辨率对数，用于逐步降低分辨率
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]  #生成所有的分辨率，从图像的原始分辨率逐步减小到 4x4
        channels_dict = {res: min(channel_base // res, channel_max) for res in
                         self.block_resolutions + [4]}  #计算每个分辨率层的通道数，通道数会随着分辨率减小而增加
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:  # 条件标签c_dim会映射到一个特征向量空间，其维度为 cmap_dim
            cmap_dim = channels_dict[4]  # 会默认等于最小分辨率下的通道数
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:  # 遍历 block_resolutions，为每个分辨率层创建一个 EncoderBlock
            in_channels = channels_dict[res] if res < img_resolution else 0  # 取决于图像的分辨率。
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]  # 通常是下一级分辨率的通道数
            use_fp16 = (res >= fp16_resolution)
            block = EncoderBlock(in_channels, tmp_channels, out_channels, resolution=res, first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)  # b4, b8, b16 等属性会被动态地创建，并存储在 self 中。
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None,
                                          **mapping_kwargs)
        self.b4 = EncoderEpilogue(channels_dict[4], cmap_dim=cmap_dim, z_dim=z_dim * 2, resolution=4, **epilogue_kwargs,
                                  **common_kwargs)

        # self.attn_layer = SelfAttention(channels_dict[256])  # 在 256x256 之后插入注意力5++++
        # self.middle = nn.Sequential(*[AOTBlock(256, 1046) for _ in range(3)])   # 6+++

    def forward(self, img, c, **block_kwargs):  #8 4 256 256
        x = None
        feats = {}
        for res in self.block_resolutions:  #x 是初始化的中间特征值，初始为 None。编码网络逐层处理输入图像，每一层的特征通过相应的 EncoderBlock 提取，并将特征存储在字典 feats 中，按分辨率作为键
            block = getattr(self, f'b{res}')
            x, img, feat = block(x, img, **block_kwargs)
            # if res == 256:  # **在 256x256 之后应用注意力**5++++++
            #     feat = self.attn_layer(feat)
            feats[res] = feat
            # 在这里插入 Feature Equalization 模块,实现它来对不同层次的特征进行均衡。
            # feat = self.feature_equalization(feat)

        cmap = None
        if self.c_dim > 0:  # 如果有条件标签c，则通过映射网络 MappingNetwork 将标签映射到一个条件特征向量 cmap
            cmap = self.mapping(None, c)

        x, const_e = self.b4(x, cmap)  #成最终的全局特征表示 x 和 4x4 特征图 const_e
        feats[4] = const_e
        # 在这里对全局特征进行 Feature Equalization,最终特征进行一次全局的均衡
        # x = self.feature_equalization(x)

        B, _ = x.shape
        z = torch.randn((B, self.z_dim), requires_grad=False, dtype=x.dtype, device=x.device)  ## Noise for Co-Modulation
        return x, z, feats


#x：全局特征表示。z：随机噪声，用于生成阶段的调制。
# ----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
                 w_dim,  # Intermediate latent (W) dimensionality.中间潜在空间的维度,W 空间是在映射网络中将输入潜在变量映射到一个更有意义的空间。
                 z_dim,  # Output Latent (Z) dimensionality.输出潜在空间的维度，通常用来定义图像生成的潜在特征
                 img_resolution,  # Output image resolution.
                 img_channels,  # Number of color channels.
                 channel_base=16384,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=0,  # Use FP16 for the N highest resolutions.
                 **block_kwargs,  # Arguments for SynthesisBlock.
                 ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0  #断言检查：确保图像分辨率是大于等于 4 且为 2 的幂次。
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))  # 计算分辨率的对数
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(3, self.img_resolution_log2 + 1)]  # 定义分辨率的序列
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
        #SynthesisForeword：用于最初的特征生成，生成 4x4 分辨率的初始图像特征。它从噪声 z_dim 以及条件信息中生成初始的低分辨率特征。
        self.foreword = SynthesisForeword(img_channels=img_channels, in_channels=min(channel_base // 4, channel_max),
                                          z_dim=z_dim * 2, resolution=4)

        self.num_ws = self.img_resolution_log2 * 2 - 2

        for res in self.block_resolutions:  #输入通道数由上一层的输出通道数决定，输出通道数通过 channels_dict 定义。
            if res // 2 in channels_dict.keys():
                in_channels = channels_dict[res // 2] if res > 4 else 0
            else:
                in_channels = min(channel_base // (res // 2), channel_max)
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)

            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                                   #SynthesisBlock：于在不同分辨率下逐层生成图像
                                   img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            setattr(self, f'b{res}', block)  #通过 setattr 方法，动态地为每个分辨率（如 b8, b16 等）创建一个 SynthesisBlock。

    def forward(self, x_global, mask, feats, ws, fname=None, **block_kwargs):  #将中间潜在特征逐步转换为图像。

        img = None
        #通过 SynthesisForeword 生成初始的低分辨率图像特征 x 和图像 img。
        x, img = self.foreword(x_global, ws, feats, img)  #ws：W空间的潜在变量，是由映射网络生成的权重集合。

        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')  #使用 getattr 动态获取每个分辨率的 SynthesisBlock，逐层对图像进行生成
            #mod_vector0, mod_vector1, mod_vector_rgb 是不同分辨率下用于调制图像特征的向量。由 W 空间的潜在变量 ws 和全局特征 x_global 组成,这些调制向量将会传递给 SynthesisBlock，用于控制生成的图像特征
            mod_vector0 = []
            mod_vector0.append(ws[:, int(np.log2(res)) * 2 - 5])
            mod_vector0.append(x_global.clone())
            mod_vector0 = torch.cat(mod_vector0, dim=1)

            mod_vector1 = []
            mod_vector1.append(ws[:, int(np.log2(res)) * 2 - 4])
            mod_vector1.append(x_global.clone())
            mod_vector1 = torch.cat(mod_vector1, dim=1)

            mod_vector_rgb = []
            mod_vector_rgb.append(ws[:, int(np.log2(res)) * 2 - 3])
            mod_vector_rgb.append(x_global.clone())
            mod_vector_rgb = torch.cat(mod_vector_rgb, dim=1)
            x, img = block(x, mask, feats, img, (mod_vector0, mod_vector1, mod_vector_rgb), fname=fname, **block_kwargs)

        return img



class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

from collections import OrderedDict

class StyleFormerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.mlp = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(d_model, d_model * 4)),
            ("act", QuickGELU()),  # 门控激活
            ("fc2", nn.Linear(d_model * 4, d_model))
        ]))

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.ln_1(x)
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        x, _ = self.attn(x, x, x, attn_mask=self.attn_mask)
        x = x + residual

        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = x + residual
        return x



# ----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):  # main
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality.# 输入的潜在向量维度 Z
                 c_dim,  # Conditioning label (C) dimensionality.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output resolution.
                 img_channels,  # Number of output color channels.
                 encoder_kwargs={},  # Arguments for EncoderNetwork.
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 synthesis_kwargs={},  # Arguments for SynthesisNetwork.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.encoder = EncoderNetwork(c_dim=c_dim, z_dim=z_dim, img_resolution=img_resolution, img_channels=img_channels, **encoder_kwargs)
        self.synthesis = SynthesisNetwork(z_dim=z_dim, w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws  # 确定映射网络 MappingNetwork 生成的风格向量个数
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        # self.reblock_attn = nn.Sequential(*[ResidualAttentionBlock(d_model=z_dim, n_head=8) for _ in range(7)])    #3++++    #每个输入会经过 6 次注意力机制的计算，每次都包括一个 MultiheadAttention 层和一个 MLP 层，同时每个 ResidualAttentionBlock 的输出都会与输入进行残差连接。
        self.reblock_attn = nn.Sequential(*[StyleFormerBlock(d_model=z_dim, n_head=8) for _ in range(6)])  # 3++++
        # self.reblock_attn_img = nn.Sequential(*[ResidualAttentionBlock(d_model=z_dim, n_head=8) for _ in range(3)])

    def forward(self, img, c, fname=None, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):

        mask = img[:, 0].unsqueeze(1)  # 提取输入图像的 mask  1 4 256 256
        x_global, z, feats = self.encoder(img, c)  # 通过编码网络获取全局特征向量、潜在噪声向量和多层次特征图
        # x_global = self.reblock_attn_img(x_global) + x_global   # global + attn + mlp
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)  # 通过映射网络将噪声向量 z 映射到中间潜在空间 W，生成风格向量 ws
        # print(ws.size())
        ws = self.reblock_attn(ws) + ws  # 3++++

        img = self.synthesis(x_global, mask, feats, ws, fname=fname, **synthesis_kwargs)  # 使用合成网络生成最终的图像

        return img


# ----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
                 c_dim,  # Conditioning label (C) dimensionality.
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
                 channel_base=16384,  # Overall multiplier for the number of channels.# 总的通道数缩放系数
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=0,  # Use FP16 for the N highest resolutions.
                 conv_clamp=None,
                 # Clamp the output of convolution layers to +-X, None = disable clamping.# 卷积层输出的值被限制在 [-X, +X]，None 表示不限制
                 cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.# 条件标签映射的维度，默认为 None
                 block_kwargs={},  # Arguments for DiscriminatorBlock.
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
                 ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]  # 定义每层的分辨率（从最高分辨率逐步降到最低）
        channels_dict = {res: min(channel_base // res, channel_max) for res in
                         self.block_resolutions + [4]}  # 定义每一层的通道数，通道数随着分辨率的降低而增加
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)  # 确定使用 FP16 的分辨率

        if cmap_dim is None:  # 条件映射的维度，如果没有指定，使用最后一层的通道数
            cmap_dim = channels_dict[4]
        if c_dim == 0:  # 如果没有条件标签，条件映射维度设为 0
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)  # 设置通用的卷积层参数
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0  # 当前层的输入通道数
            tmp_channels = channels_dict[res]  # 当前层的中间通道数
            out_channels = channels_dict[res // 2]  # 当前层的输出通道数
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                                       #用于逐步下采样图像并提取特征。每个块处理输入图像的一个分辨率层。
                                       first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs,
                                       **common_kwargs)
            setattr(self, f'b{res}', block)  #动态创建每一层的判别器块
            cur_layer_idx += block.num_layers  # 更新当前层的索引
        if c_dim > 0:  #将输入的条件标签（如类别信息）映射到一个特定的向量空间，这个向量用于帮助判别器进行条件判断。
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None,
                                          **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs,
                                        # 初始化最后的判别器末端模块,将从每层提取的特征汇总，并生成最终的判别结果
                                        **common_kwargs)

    def forward(self, img, c, **block_kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')  # 获取相应分辨率的判别器块
            x, img = block(x, img, **block_kwargs)  # 传入当前特征图和图像，通过卷积层更新特征

        cmap = None
        if self.c_dim > 0:  # 如果有条件标签，生成条件映射向量
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)  # 通过判别器末端模块，生成最终的输出
        return x

# -----------------------------------------------------
# -----------------------
