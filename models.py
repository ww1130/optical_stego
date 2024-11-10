import os
import numpy as np
#import cv2
#import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import math


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


# class Encoder(nn.Module):
#     def __init__(self, host_channels=3, secret_channels=1, embed_channels=64):
#         super(Encoder, self).__init__()
#
#         # 第一层：融合host和secret信息
#         self.conv1 = nn.Conv3d(host_channels + secret_channels, embed_channels, kernel_size=3, padding=1)
#         # self.relu = nn.ReLU(inplace=True)
#         # self.leakyrelu = nn.LeakyReLU(inplace=True)
#         self.relu = nn.ReLU(inplace=False)
#         self.leakyrelu = nn.LeakyReLU(inplace=False)
#         # 中间层：多层卷积进行特征提取和嵌入
#         self.conv2 = nn.Conv3d(embed_channels, embed_channels, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv3d(embed_channels, embed_channels, kernel_size=3, padding=1)
#
#
#         # 输出层：恢复到host的通道数
#         self.conv_out = nn.Conv3d(embed_channels, host_channels, kernel_size=3, padding=1)
#         self.activation = nn.Tanh()  # 可以根据需求选择激活函数，如 Tanh 或 Sigmoid
#
#     def forward(self,host,secret):
#         # host = host.requires_grad_(True)
#         # secret = secret.requires_grad_(True)
#         host_1=host[:,1:,:,:,:,:]
#         # 将输入转换为 (b, c, t, 4, h/2, w/2) 以适应Conv3D
#         host_1 = host_1.permute(0, 2, 1, 3, 4, 5)  # (b, c, t, 4, h/2, w/2)
#         secret = secret.permute(0, 2, 1, 3, 4, 5)  # (b, 1, t, 4, h/2, w/2)
#
#         b, c, t, sub, h_half, w_half = host_1.shape
#         host_1 = host_1.reshape(b, c, t * sub, h_half, w_half)  # (b, c, t*4, h/2, w/2)
#         secret = secret.reshape(b, 1, t * sub, h_half, w_half)  # (b, 1, t*4, h/2, w/2)
#
#         x = torch.cat([host_1, secret], dim=1)
#
#         #x.requires_grad_(True)
#         #stego.requires_grad_(True)
#         # 编码过程
#         x = self.leakyrelu(self.conv1(x))  # (b, embed_channels, t, 4, h/2, w/2)
#         x = self.leakyrelu(self.conv2(x))  # (b, embed_channels, t, 4, h/2, w/2)
#         x = self.leakyrelu(self.conv3(x))  # (b, embed_channels, t, 4, h/2, w/2)
#         stego = self.activation(self.conv_out(x)) # (b, c, t, 4, h/2, w/2)
#
#
#         stego=stego.reshape(b, c, t, sub, h_half, w_half)
#         stego = stego.permute(0, 2, 1, 3, 4, 5)
#         host_1 = host_1.reshape(b, c, t , sub, h_half, w_half)
#         host_1 = host_1.permute(0, 2, 1, 3, 4, 5)
#
#         stego = stego + host_1
#
#
#         return stego


class DenseEncoder(nn.Module):
    """
    The DenseEncoder3D module takes a cover video tensor and a data tensor,
    and combines them into a steganographic video.
    Input: cover (B, T, C, 4, H/2, W/2), secret (B, T, 1, 4, H/2, W/2)
    Output: steganographic video (B, T, C, 4, H/2, W/2)
    """
    def __init__(self, data_depth=1, hidden_size=64):
        super(DenseEncoder, self).__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self._build_models()
    def _conv3d(self, in_channels, out_channels):
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_models(self):
        # Initialize the layers for the encoder with concatenated channel and frequency dimensions
        in_channels = 3 * (4-1)  # Assuming input channels and DCT dimension are combined

        self.features = nn.Sequential(
            self._conv3d(in_channels, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(self.hidden_size),
        )

        self.conv1 = nn.Sequential(
            self._conv3d(self.hidden_size + self.data_depth * 4, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(self.hidden_size),
        )
        self.conv2 = nn.Sequential(
            self._conv3d(self.hidden_size*2 + self.data_depth * 4, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(self.hidden_size),
        )
        self.conv3 = nn.Sequential(
            self._conv3d(self.hidden_size * 3 + self.data_depth * 4, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(self.hidden_size),
        )
        self.conv4 = nn.Sequential(
            self._conv3d(self.hidden_size * 4 + self.data_depth * 4, in_channels)
        )

        # self.conv1 = nn.Sequential(
        #     self._conv3d(in_channels, self.hidden_size),
        #     nn.LeakyReLU(inplace=True),
        #     nn.BatchNorm3d(self.hidden_size),
        # )
        # self.conv2 = nn.Sequential(
        #     self._conv3d(self.hidden_size + self.data_depth * 4, self.hidden_size),
        #     nn.LeakyReLU(inplace=True),
        #     nn.BatchNorm3d(self.hidden_size),
        # )
        # self.conv3 = nn.Sequential(
        #     self._conv3d(self.hidden_size * 2 + self.data_depth * 4, self.hidden_size),
        #     nn.LeakyReLU(inplace=True),
        #     nn.BatchNorm3d(self.hidden_size),
        # )
        # self.conv4 = nn.Sequential(
        #     self._conv3d(self.hidden_size * 3 + self.data_depth * 4, in_channels)
        # )

        # Store the convolutional layers in a list
        #self._models = [self.conv1, self.conv2, self.conv3, self.conv4]
        self._models = [self.features,self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, video, secret):
        """
        Forward pass for DenseEncoder3D.
        Input:
            video: (B, T, C, 4, H/2, W/2)
            secret: (B, T, 1, 4, H/2, W/2)
        Output:
            steganographic video: (B, T, C, 4, H/2, W/2)
        """
        # Combine C and 4 dimensions by reshaping
        b, t, c, d, h, w = video.size()
        video = video[:,1 :, :, :, :,:]
        high_band = video[:, :, :, 1:, :, :]
        low_band = video[:, :, :, :1, :, :]
        high_band = high_band.reshape(b, t-1, c * 3, h, w)
        high_band = high_band.permute(0, 2, 1, 3, 4)

        secret = secret.reshape(b, t-1 , self.data_depth * 4, h, w)
        secret = secret.permute(0, 2, 1, 3, 4)

        # Apply the first convolutional layer
        x = self._models[0](high_band)
        x_list = [x]

        # Loop over the remaining layers and concatenate `secret` to each layer
        for layer in self._models[1:]:
            x = layer(torch.cat(x_list + [secret], dim=1))
            x_list.append(x)

        high_band = high_band + x

        high_band = high_band.permute(0, 2, 1, 3, 4).reshape(b, t-1, c, 3, h, w)
        stego_video = torch.cat((low_band, high_band), dim=3)

        # Reshape back to the original dimensions (B, T, C, 4, H/2, W/2)
        return stego_video

# class Decoder(nn.Module):
#     def __init__(self, host_channels=3, secret_channels=1, embed_channels=64):
#         super(Decoder, self).__init__()
#         # 第一层卷积
#         self.conv1 = nn.Conv3d(host_channels, embed_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm3d(embed_channels)  # 批量归一化
#         # self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.ReLU(inplace=False)
#         # 第二层卷积
#         self.conv2 = nn.Conv3d(embed_channels, embed_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm3d(embed_channels)  # 批量归一化
#
#         # 第三层卷积
#         self.conv3 = nn.Conv3d(embed_channels, embed_channels * 2, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm3d(embed_channels * 2)  # 批量归一化
#
#         # 反卷积层（上采样）: 恢复空间分辨率
#         self.deconv1 = nn.ConvTranspose3d(embed_channels * 2, embed_channels, kernel_size=3, padding=1,)
#         self.bn4 = nn.BatchNorm3d(embed_channels)
#
#         self.deconv_out = nn.ConvTranspose3d(embed_channels, secret_channels, kernel_size=3,  padding=1,)
#         self.activation = nn.Sigmoid()
#
#     def forward(self, stego):
#         # stego: (b, 1, t, 4, h/2, w/2)
#         stego = stego.permute(0, 2, 1, 3, 4, 5)  # (b, 1, t, 4, h/2, w/2)
#         b, c, t, sub, h_half, w_half = stego.shape
#         # 重排形状：变成 (b, c, t*4, h/2, w/2)
#         stego = stego.reshape(b, c, t * sub, h_half, w_half)
#         # 第一层卷积 + 批量归一化 + 激活
#         x = self.relu(self.bn1(self.conv1(stego)))
#         # 第二层卷积 + 批量归一化 + 激活
#         x = self.relu(self.bn2(self.conv2(x)))
#         # 第三层卷积 + 批量归一化 + 激活
#         x = self.relu(self.bn3(self.conv3(x)))
#         # 反卷积层进行上采样
#         x = self.relu(self.bn4(self.deconv1(x)))
#         # 输出层：通过反卷积恢复原始空间分辨率
#         secret_hat = self.deconv_out(x)
#         # 恢复到原始形状： (b, 1, t, 1, h, w)
#         secret_hat = secret_hat.reshape(b, 1, t, sub, h_half , w_half )  # 由于上采样，恢复到原始大小
#         secret_hat = secret_hat.permute(0, 2, 1, 3, 4, 5)  # (b, t, 1, sub, h, w)
#
#         return secret_hat

class DenseDecoder(nn.Module):
    """
    The DenseDecoder module takes a steganographic video (3D data) and attempts to decode
    the embedded data tensor.
    Input: (B, T, C, H, W)
    Output: (B, T, 1, H, W)
    """
    def __init__(self, data_depth=1, hidden_size=64):
        super(DenseDecoder, self).__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self._build_models()
    def _conv3d(self, in_channels, out_channels):
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
    def _build_models(self):
        # Define each layer as a sequential 3D convolution
        self.conv1 = nn.Sequential(
            self._conv3d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(self.hidden_size)
        )
        self.conv2 = nn.Sequential(
            self._conv3d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(self.hidden_size)
        )
        self.conv3 = nn.Sequential(
            self._conv3d(self.hidden_size * 2, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(self.hidden_size)
        )
        self.conv4 = nn.Sequential(self._conv3d(self.hidden_size * 3, self.data_depth))
        # Store each convolutional block
        self._models = [self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, x):
        """
        Forward pass for DenseDecoder.
        Input shape: (B, T, C, H, W)
        Output shape: (B, T, 1, H, W)
        """
        # Permute to (B, C, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        # Initialize first convolution
        x = self._models[0](x)
        if len(self._models) > 1:
            x_list = [x]
            for layer in self._models[1:]:
                # Concatenate previous outputs across channels before each layer
                x = layer(torch.cat(x_list, dim=1))
                x_list.append(x)

        # Output shape should be (B, 1, T, H, W); permute to (B, T, 1, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return x


class StegoLoss(nn.Module):
    def __init__(self):
        super(StegoLoss, self).__init__()

    #def forward(self, originalHavefirstframe, stego, extracted_secret, original_secret):
    def forward(self, originalHavefirstframe, stego):
        original =originalHavefirstframe[:,1:,:,:,:]
        mse = torch.mean((original - stego) ** 2)
        # if mse == 0:
        #     return float('inf')
        # psnr_loss=20 * math.log10(255.0 / math.sqrt(psnr_loss))

        # # 计算比特准确率
        # correct_bits = (extracted_secret == original_secret).float().sum()
        # total_bits = original_secret.numel()  # numel() 返回张量的元素总数
        # bitwise_accuracy = correct_bits / total_bits
        # # 如果您想要将比特准确率也作为一个损失返回，可以如下操作：
        # bitwise_loss = 1 - bitwise_accuracy  # 这会将准确率转换为损失

        #msebit=torch.mean((extracted_secret-original_secret)**2)
        #bitwise_loss=msebit

        # bitwise_loss = torch.mean((extracted_secret - original_secret) ** 2)

        return mse #, bitwise_loss

class StegoLosstest(nn.Module):
    def __init__(self):
        super(StegoLosstest, self).__init__()

    def forward(self, originalHavefirstframe, stego, extracted_secret, original_secret):
        original = originalHavefirstframe[:, 1:, :, :, :]
        mse = torch.mean((original - stego) ** 2)
            # if mse == 0:
            #     return float('inf')
            # psnr_loss=20 * math.log10(255.0 / math.sqrt(psnr_loss))

            # 计算比特准确率
        # correct_bits = (extracted_secret == original_secret).float().sum()
        # total_bits = original_secret.numel()  # numel() 返回张量的元素总数
        # bitwise_accuracy = correct_bits / total_bits
        correct_bits = (extracted_secret == original_secret).float()  # 比较相同的比特，得到一个 0 和 1 的 tensor
        bit_accuracy = correct_bits.mean()  # 计算准确率，所有正确比特的比例

        return mse, bit_accuracy



