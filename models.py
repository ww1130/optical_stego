import os
import numpy as np
import cv2
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class ChannelAttention(nn.Module):
    def __init__(self, in_channel=3, hidden_channel=1, out_channel=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.add_pooling = lambda x: self.avg_pool(x) + self.max_pool(x)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=hidden_channel,
                               kernel_size=(1, 1))  # 依据实际情况可以调整为 线性层或者一维卷积,下同
        self.conv2 = nn.Conv2d(in_channels=hidden_channel, out_channels=out_channel, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        B, T, C, H, W = x.size()
        x_reshaped = x.reshape(-1, C, H, W)
        x_pre = self.add_pooling(x_reshaped)  # (64, 32, 1, 1)
        att = self.sigmoid(self.conv2(self.relu(self.conv1(x_pre))))  # (64, 32, 1, 1)
        att = att.reshape(B, T, C, 1, 1).expand(B,T,C,H,W)
        return att


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.max_pooling = lambda x: torch.max(x, dim=1, keepdim=True)[0]
        self.ave_pooling = lambda x: torch.mean(x, dim=1, keepdim=True)
        self.conv = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B,T,C,H,W=x.size()
        x_reshaped = x.reshape(-1, C, H, W)
        att = self.sigmoid(
            self.conv(torch.cat([self.max_pooling(x_reshaped), self.ave_pooling(x_reshaped)], dim=1)))  # (64, 1, 18, 100)
        att = att.reshape(B, T, 1, H, W).expand(B,T,C,H,W)
        return att


class TemporalAttention(nn.Module):
    def __init__(self, num_temporal_features=6):
        super(TemporalAttention, self).__init__()
        self.fc1 = nn.Linear(num_temporal_features, num_temporal_features // 8, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(num_temporal_features // 8, num_temporal_features, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=2, keepdim=True)
        max_out, _ = torch.max(x, dim=2, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=2)
        x = self.fc2(self.relu1(self.fc1(x)))
        return self.sigmoid(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.channel_attentionll = ChannelAttention()
        self.channel_attentionlh = ChannelAttention()
        self.channel_attentionhl = ChannelAttention()
        self.channel_attentionhh = ChannelAttention()

        self.spatial_attentionll = SpatialAttention()
        self.spatial_attentionlh = SpatialAttention()
        self.spatial_attentionhl = SpatialAttention()
        self.spatial_attentionhh = SpatialAttention()

        self.temporal_attention = TemporalAttention()
        #self.band_attention = BandAttention()
    # def forward(self,host,secret):
    #     host_sliced=host[:,1:,:,:,:]
    #
    #     spatial_attention = self.spatial_attention(host_sliced)  # 空间注意力
    #     channel_attention = self.channel_attention(host_sliced)  # 通道注意力
    #
    #     #temporal_attention = self.temporal_attention(host_sliced)  # 时间注意力
    #
    #
    #     # 使用 host_dwt 的注意力信息调整 secret_dwt
    #     #secret_expanded = secret.expand(-1, -1, 3, -1, -1)
    #     adjusted_secret_dwt = secret * channel_attention * spatial_attention #* temporal_attention
    #     stego_dwt = secret.expand(-1, -1, 3, -1, -1) + host_sliced
    #     return stego_dwt  # 返回调整后的 stego_dwt
    def forward(self,hll,hlh,hhl,hhh,sll,slh,shl,shh):
        hll_sliced = hll[:,1:,:,:,:]
        hlh_sliced = hlh[:, 1:, :, :, :]
        hhl_sliced = hhl[:, 1:, :, :, :]
        hhh_sliced = hhh[:, 1:, :, :, :]

        spatial_attentionll = self.spatial_attentionll(hll_sliced)  # 空间注意力
        spatial_attentionlh = self.spatial_attentionlh(hlh_sliced)  # 空间注意力
        spatial_attentionhl = self.spatial_attentionhl(hhl_sliced)  # 空间注意力
        spatial_attentionhh = self.spatial_attentionhh(hhh_sliced)  # 空间注意力

        channel_attentionll = self.channel_attentionll(hll_sliced)  # 空间注意力
        channel_attentionlh = self.channel_attentionlh(hlh_sliced)  # 空间注意力
        channel_attentionhl = self.channel_attentionhl(hhl_sliced)  # 空间注意力
        channel_attentionhh = self.channel_attentionhh(hhh_sliced)  # 空间注意力

        #temporal_attention = self.temporal_attention(host_sliced)  # 时间注意力


        # 使用 host_dwt 的注意力信息调整 secret_dwt
        #secret_expanded = secret.expand(-1, -1, 3, -1, -1)
        adjusted_secret_ll = sll * channel_attentionll * spatial_attentionll #* temporal_attention
        adjusted_secret_ll = sll * channel_attentionll * spatial_attentionll  # * temporal_attention
        adjusted_secret_ll = sll * channel_attentionll * spatial_attentionll  # * temporal_attention
        adjusted_secret_ll = sll * channel_attentionll * spatial_attentionll  # * temporal_attention

        stegoll = sll.expand(-1, -1, 3, -1, -1) + hll_sliced
        stegolh = slh.expand(-1, -1, 3, -1, -1) + hlh_sliced
        stegohl = shl.expand(-1, -1, 3, -1, -1) + hhl_sliced
        stegohh = shh.expand(-1, -1, 3, -1, -1) + hhh_sliced

        return stegoll,stegolh,stegohl,stegohh  # 返回调整后的 stego_dwt


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # 假设空间注意力、通道注意力和时间注意力与编码器中的结构相同
        self.channel_attention = ChannelAttention()
        self.spatial_attention = SpatialAttention()
        self.temporal_attention = TemporalAttention()

        # 添加用于恢复 host 和 secret 的层
        self.host_recovery = nn.Conv2d(3, 3, kernel_size=1)  # 假设输入和输出都是三通道
        self.secret_recovery = nn.Conv2d(3, 1, kernel_size=1)  # 假设 secret 是单通道

    def forward(self, stego_dwt):
        # 首先，我们需要从 stego_dwt 中分离出 host 和 secret 的部分
        # 这里假设 stego_dwt 是通过将 secret 扩展到 host 的通道数然后相加得到的
        secret_expanded = stego_dwt[:, :, :3, :, :]  # 假设 secret 被扩展到了前三个通道
        host_sliced = stego_dwt[:, :, 3:, :, :]  # 假设 host 在后三个通道

        # 应用注意力机制来增强恢复过程
        channel_attention = self.channel_attention(stego_dwt)
        spatial_attention = self.spatial_attention(stego_dwt)

        # 使用注意力信息调整 secret_expanded
        adjusted_secret = secret_expanded * channel_attention * spatial_attention

        # 恢复 host 和 secret
        recovered_host = self.host_recovery(host_sliced)
        recovered_secret = self.secret_recovery(adjusted_secret)

        # 返回恢复的 host 和 secret
        return recovered_host, recovered_secret


