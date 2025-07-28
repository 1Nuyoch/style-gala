# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:40:54 2024

@author: ZY
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.append("/media/ww/软件/FcF-Inpainting/training/Gsop/")
import MPNCOV

def cov_feature(x):
    # x.shape = (B,C,8,8)
    batchsize = x.data.shape[0]
    dim = x.data.shape[1]
    h = x.data.shape[2]
    w = x.data.shape[3]
    M = h * w
    x = x.reshape(batchsize, dim, M)
    I_hat = (-1. / M / M) * torch.ones(dim, dim, device=x.device) + (1. / M) * torch.eye(dim, dim, device=x.device)
    I_hat = I_hat.view(1, dim, dim).repeat(batchsize, 1, 1).type(x.dtype)
    y = (x.transpose(1, 2)).bmm(I_hat).bmm(x)
    return y


class GSoP(nn.Module):

    def __init__(self, indim, attention='0', att_dim=128):
        super(GSoP, self).__init__()
        self.dimDR = att_dim   # 128
        self.relu = nn.ReLU(inplace=True)
        self.relu_normal = nn.ReLU(inplace=False)
        if attention in {'1', '+', 'M', '&'}:
            # if planes > 64:
            #     DR_stride=1
            # else:
            #     DR_stride=2
            self.ch_dim = att_dim
            self.conv_for_DR = nn.Conv2d(indim, self.ch_dim, kernel_size=1, stride=1, bias=True)
            self.bn_for_DR = nn.BatchNorm2d(self.ch_dim)
            self.row_bn = nn.BatchNorm2d(self.ch_dim)
            # row-wise conv is realized by group conv
            self.row_conv_group = nn.Conv2d(self.ch_dim, 4 * self.ch_dim, kernel_size=(self.ch_dim, 1),
                                            groups=self.ch_dim, bias=True)
            self.fc_adapt_channels = nn.Conv2d(4 * self.ch_dim, indim, kernel_size=1, groups=1, bias=True)
            self.sigmoid = nn.Sigmoid()

        if attention in {'2', '+', 'M', '&'}:
            self.sp_d = att_dim    # 128
            self.sp_h = 8
            self.sp_w = 8
            self.sp_reso = self.sp_h * self.sp_w  # 64
            self.conv_for_DR_spatial = nn.Conv2d(indim, self.sp_d, kernel_size=1, stride=1, bias=True)
            self.bn_for_DR_spatial = nn.BatchNorm2d(self.sp_d)

            self.adppool = nn.AdaptiveAvgPool2d((self.sp_h, self.sp_w))
            self.row_bn_for_spatial = nn.BatchNorm2d(self.sp_reso)
            # row-wise conv is realized by group conv
            self.row_conv_group_for_spatial = nn.Conv2d(self.sp_reso, self.sp_reso * 4, kernel_size=(self.sp_reso, 1),
                                                        groups=self.sp_reso, bias=True)
            self.fc_adapt_channels_for_spatial = nn.Conv2d(self.sp_reso * 4, self.sp_reso, kernel_size=1, groups=1, bias=True)
            self.sigmoid = nn.Sigmoid()
            self.adpunpool = F.adaptive_avg_pool2d

        if attention == '&':  # we employ a weighted spatial concat to keep dim
            self.groups_base = 32
            self.groups = int(indim / 64)
            self.factor = int(math.log(self.groups_base / self.groups, 2))
            self.padding_num = self.factor + 2
            self.conv_kernel_size = self.factor * 2 + 5
            self.dilate_conv_for_concat1 = nn.Conv2d(indim,indim, kernel_size=(self.conv_kernel_size, 1),
                                                     stride=1, padding=(self.padding_num, 0),
                                                     groups=self.groups, bias=True)
            self.dilate_conv_for_concat2 = nn.Conv2d(indim, indim, kernel_size=(self.conv_kernel_size, 1),
                                                     stride=1, padding=(self.padding_num, 0),
                                                     groups=self.groups, bias=True)
            self.bn_for_concat = nn.BatchNorm2d(indim)

        self.attention = attention

    def chan_att(self, out):
        # NxCxHxW

        out = self.relu_normal(out)
        out = self.conv_for_DR(out)  # down channel
        out = self.bn_for_DR(out)
        out = self.relu(out)  # NxCxHxW

        out = MPNCOV.CovpoolLayer(out)  # Nxdxd
        out = out.view(out.size(0), out.size(1), out.size(2), 1).contiguous()  # Nxdxdx1

        out = self.row_bn(out)
        out = self.row_conv_group(out)  # Nx512x1x1

        out = self.fc_adapt_channels(out)  # NxCx1x1
        out = self.sigmoid(out)  # NxCx1x1

        return out

    def pos_att(self, out):
        pre_att = out  # NxCxHxW
        out = self.relu_normal(out)
        out = self.conv_for_DR_spatial(out)  # down channel
        out = self.bn_for_DR_spatial(out)

        out = self.adppool(out)  # keep the feature map size to 8x8  NxCx8x8

        out = cov_feature(out)  # Nx64x64
        out = out.view(out.size(0), out.size(1), out.size(2), 1).contiguous()  # Nx64x64x1
        out = self.row_bn_for_spatial(out)

        out = self.row_conv_group_for_spatial(out)  # Nx256x1x1
        out = self.relu(out)

        out = self.fc_adapt_channels_for_spatial(out)  # Nx64x1x1
        out = self.sigmoid(out)
        out = out.view(out.size(0), 1, self.sp_h, self.sp_w).contiguous()  # Nx1x8x8

        out = self.adpunpool(out, (pre_att.size(2), pre_att.size(3)))  # unpool Nx1xHxW

        return out

    def forward(self, x):

        out = x  # x.shape = (B,C,H,W)
        out = out.float()
        if self.attention == '1':  # channel attention,GSoP default mode
            pre_att = out
            att = self.chan_att(out)
            out = pre_att * att

        elif self.attention == '2':  # position attention
            pre_att = out
            att = self.pos_att(out)
            out = self.relu_normal(pre_att * att)

        elif self.attention == '+':  # fusion manner: average
            pre_att = out
            chan_att = self.chan_att(out)
            pos_att = self.pos_att(out)
            out = pre_att * chan_att + self.relu(pre_att.clone() * pos_att)

        elif self.attention == 'M':  # fusion manner: MAX
            pre_att = out  # 64 768 7 7
            chan_att = self.chan_att(out)  # 64 768 1 1
            pos_att = self.pos_att(out)     # 64 1 7 7
            out = torch.max(pre_att * chan_att, self.relu(pre_att.clone() * pos_att))  # 64 768 7 7 all

        elif self.attention == '&':  # fusion manner: concat
            pre_att = out
            chan_att = self.chan_att(out)
            pos_att = self.pos_att(out)
            out1 = self.dilate_conv_for_concat1(pre_att * chan_att)
            out2 = self.dilate_conv_for_concat2(self.relu(pre_att * pos_att))
            out = out1 + out2
            out = self.bn_for_concat(out)

        return out
