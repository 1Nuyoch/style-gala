# mdta.py多头转置自注意力
import torch
import torch.nn as nn
from einops import rearrange
# nn.MultiheadAttention(d_model, n_head)
# class MDTA(nn.Module):
#     def __init__(self, dim, num_heads, bias=True, reduction=2):
#         super(MDTA, self).__init__()
#
#         # 假设 tmp_channels 应等于 dim，因为没有其他提供的参数
#         tmp_channels = dim
#
#         # 定义 qkv 卷积层
#         self.qkv = nn.Conv2d(tmp_channels, tmp_channels * 3, kernel_size=1, bias=bias)
#
#         # 定义 qkv_dwconv 和 project_out，确保在 forward 中使用前它们已定义
#         self.qkv_dwconv = nn.Conv2d(tmp_channels * 3, tmp_channels * 3, kernel_size=3, padding=1,
#                                     groups=tmp_channels * 3, bias=bias)
#         self.project_out = nn.Conv2d(tmp_channels, tmp_channels, kernel_size=1, bias=bias)
#
#         # 设置其他参数
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1, dtype=torch.float16))  # 使用 float16
#
#         # 位置编码参数
#         reduced_dim = dim  # 保持 reduced_dim 等于 dim 本身，确保一致性
#         self.position_encoding = nn.Parameter(
#             torch.randn(1, num_heads, dim // num_heads, 1, dtype=torch.float16),
#             requires_grad=True
#         )
#
#         # 转换所有模块到 float16
#         self.qkv = self.qkv.to(torch.float16)
#         self.qkv_dwconv = self.qkv_dwconv.to(torch.float16)
#         self.project_out = self.project_out.to(torch.float16)
#
#     def forward(self, x):
#         x = x.to(torch.float16)  # 确保输入是 float16
#         b, c, h, w = x.shape
#
#         # 计算 qkv 并分割
#         qkv = self.qkv_dwconv(self.qkv(x))
#         q, k, v = qkv.chunk(3, dim=1)
#
#         # 重排列 q, k, v
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         # 应用位置编码并规范化
#         q = torch.nn.functional.normalize(q + self.position_encoding, dim=-1)
#         k = torch.nn.functional.normalize(k + self.position_encoding, dim=-1)
#
#         # 计算注意力
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#
#         # 应用注意力到 v
#         out = (attn @ v)
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#         # 投影输出
#         out = self.project_out(out)
#         return out


class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # 确保 x 的类型为 float，并与其他层保持一致
        dtype = x.dtype  # 获取输入的 dtype  float 16

        x = x.to(dtype)  # 保持输入的 dtype 与其他层一致

        self.qkv = self.qkv.to(dtype)
        self.qkv_dwconv = self.qkv_dwconv.to(dtype)
        self.project_out = self.project_out.to(dtype)

        # 获取形状
        b, c, h, w = x.shape
        # 计算 qkv
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        # 进行 q, k, v 的形状重排
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # 归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.temperature.to(dtype)  # 确保 self.temperature 与 x 的 dtype 一致
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out