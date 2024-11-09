import os
import numpy as np
#import cv2
#import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image,ImageDraw
import models

def dwt_transform(tensor):
    # 输入 tensor 形状为 (B, T, C, H, W)
    B, T, C, H, W = tensor.shape
    dwt_coeffs = []

    # 对每个 batch、时间步、通道分别进行 DWT 变换
    for b in range(B):
        batch_coeffs = []
        for t in range(T):
            time_coeffs = []
            for c in range(C):
                # 获取 (H, W) 的二维图像数据

                img = tensor[b, t, c, :, :].cpu().numpy()
                # 使用 pywt 进行 DWT 变换
                coeffs = pywt.dwt2(img, 'haar')
                ll, (lh, hl, hh) = coeffs
                # 将结果转换为 torch 张量

                time_coeffs.append((torch.tensor(ll), torch.tensor(lh), torch.tensor(hl), torch.tensor(hh)))
            batch_coeffs.append(time_coeffs)
        dwt_coeffs.append(batch_coeffs)

    return dwt_coeffs

def dwt_inverse(dwt_coeffs):
    B = len(dwt_coeffs)
    T = len(dwt_coeffs[0])
    C = len(dwt_coeffs[0][0])

    # 重建原始图像
    reconstructed_images = []

    for b in range(B):
        batch_imgs = []
        for t in range(T):
            time_imgs = []
            for c in range(C):
                ll, lh, hl, hh = dwt_coeffs[b][t][c]

                # 使用 pywt 进行逆 DWT 变换
                img_reconstructed = pywt.idwt2((ll.detach().cpu().numpy(), (lh.detach().cpu().numpy(), hl.detach().cpu().numpy(), hh.detach().cpu().numpy())), 'haar')

                # 将结果转换回张量，并确保保留梯度信息
                img_reconstructed_tensor = torch.tensor(img_reconstructed, dtype=torch.float32, device=ll.device,requires_grad=True)
                #img_reconstructed_tensor.requires_grad_(True)

                time_imgs.append(img_reconstructed_tensor)
            batch_imgs.append(torch.stack(time_imgs, dim=0))  # (C, H, W)
        reconstructed_images.append(torch.stack(batch_imgs, dim=0))  # (T, C, H, W)

    return torch.stack(reconstructed_images, dim=0)  # (B, T, C, H, W)


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


def dwt_init(x):
    # 假设 x 的形状是 (b, t, c, h, w)

    # 按照行进行分解
    x01 = x[:, :, :, 0::2, :] / 2  # 偶数行
    x02 = x[:, :, :, 1::2, :] / 2  # 奇数行

    # 按照列进行分解
    x1 = x01[:, :, :, :,0::2]  # 偶数列
    x2 = x02[:, :, :, :, 0::2]  # 奇数列
    x3 = x01[:, :, :, :,1::2]  # 偶数列
    x4 = x02[:, :, :, :,1::2]  # 奇数列

    # 计算四个频带
    x_LL = x1 + x2 + x3 + x4  # 低频（低低频）
    x_HL = -x1 - x2 + x3 + x4  # 高频水平（低高频）
    x_LH = -x1 + x2 - x3 + x4  # 高频垂直（高低频）
    x_HH = x1 - x2 - x3 + x4  # 高频对角线（高高频）

    x_LL=x_LL.unsqueeze(5)
    x_HL = x_HL.unsqueeze(5)
    x_LH = x_LH.unsqueeze(5)
    x_HH = x_HH.unsqueeze(5)


    stacked=torch.cat((x_LL, x_HL, x_LH, x_HH), dim=5)
    stacked=stacked.permute(0, 1, 2, 5, 3, 4)
    # 将四个频带沿 channel 维度拼接
    return stacked  # 拼接在 c 维度，结果形状为 (b, t, c, 4, h/2, w/2)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


def iwt_init(x):
    r = 2  # 缩放因子（在每个方向上扩大尺寸）
    in_batch, in_time, in_channel, in_freq, in_height, in_width = x.size()

    # 计算输出的通道数和图像尺寸
    out_batch, out_time, out_channel, out_height, out_width = in_batch, in_time, in_channel , r * in_height, r * in_width

    # 获取每个频带
    x1 = x[:, :, :, 0, :, :] / 2  # 第一个频带
    x2 = x[:, :, :, 1, :, :] / 2  # 第二个频带
    x3 = x[:, :, :, 2, :, :] / 2  # 第三个频带
    x4 = x[:, :, :, 3, :, :] / 2  # 第四个频带

    # 初始化输出张量
    h = torch.zeros([out_batch, out_time, out_channel, out_height, out_width]).float().to(x.device)

    # 使用 IWT 的逆变换公式，将4个频带合并成一个输出图像
    h[:, :, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class StegoTensorProcessor:
    def __init__(self, stego_dwt):
        """
        初始化 StegoTensorProcessor 类。
        参数:
        stego_dwt (torch.Tensor): 形状为 (B, T, C, 4, H, W) 的张量。
        """
        self.stego_dwt = stego_dwt

    def process(self):
        """
        处理张量，将其转换成嵌套列表结构。
        返回:
        result_list (list): 嵌套列表结构，其中 B, T, C 这几个维度转换成列表，4 转换成元组，H 和 W 仍然是二维张量。
        """
        B, T, C, _, H, W = self.stego_dwt.size()
        result_list = []
        for b in range(B):
            batch_list = []
            for t in range(T):
                temporal_list = []
                for c in range(C):
                    channel_list = []
                    for b_index in range(4):  # 频带维度
                        band_tensor = self.stego_dwt[b, t, c, b_index, :, :]
                        channel_list.append(band_tensor)
                    temporal_list.append(tuple(channel_list))
                batch_list.append(temporal_list)
            result_list.append(batch_list)

        return result_list

def convert_to_tensor(item):
    if isinstance(item, torch.Tensor):
        return item
    elif isinstance(item, (list, tuple)):
        return torch.stack([convert_to_tensor(subitem) for subitem in item])
    else:
        return torch.tensor(item)


def add_noise(image, flow_gray):
    # 确保输入图像的形状是三维的 (H, W, C)
    if len(image.shape) != 3:
        raise ValueError("Input image must have shape (H, W, C)")

    block_size = 16
    height, width, channels = image.shape
    noisy_image = image.copy()

    # 对每个图像块叠加噪声
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # 取出灰度图像对应位置的块
            block = flow_gray[0, i:i + block_size, j:j + block_size]  # 灰度图块的大小 (block_size, block_size)
            # 跳过图像边缘块
            if block.shape[0] != block_size or block.shape[1] != block_size:
                continue
            # 计算块亮度均值
            avg_brightness = np.mean(block)
            noise_intensity = avg_brightness / 255.0  # 亮度噪声
            # 创建与块匹配的噪声
            noise = np.random.normal(scale=noise_intensity * 25, size=(block_size, block_size, channels))
            # 确保噪声形状与图像块形状匹配
            noisy_image[i:i + block_size, j:j + block_size, :] += noise
    # 像素值
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return image
    #return noisy_image


def add_noise_based_on_variance(stego_image, flow_gray, block_size=(8, 8), max_noise=0.1):
    B, T, C, H, W = stego_image.shape  # 正确读取尺寸
    noise_image = stego_image.clone()  # 复制隐写图像以添加噪声

    flow_gray_reshaped = flow_gray.view(B * T, 1, H, W)

    for b in range(B):
        for t in range(T):
            flow_gray_current = flow_gray_reshaped[b * T + t]

            # 将当前帧的灰度图转换为PIL图像以便绘制
            flow_gray_pil = Image.fromarray((flow_gray_current.squeeze().cpu().numpy() * 255).astype('uint8'), 'L')
            draw = ImageDraw.Draw(flow_gray_pil)

            # 分块处理
            for h in range(0, H, block_size[0]):
                for w in range(0, W, block_size[1]):
                    block = flow_gray_current[:, h:h + block_size[0], w:w + block_size[1]]
                    variance = torch.var(block)

                    # 根据方差生成噪声
                    noise_level = max_noise * variance.item()
                    noise = torch.randn(C, block_size[0], block_size[1], device=stego_image.device) * noise_level

                    # 将噪声添加到隐写图像中
                    noise_image[b, t, :, h:h + block_size[0], w:w + block_size[1]] += noise

                    # 绘制分块边界
                    draw.rectangle([w, h, w + block_size[1], h + block_size[0]], outline='red', width=1)

            # 保存带有划分线的灰度图
            #filename = f'block_divided_flow_gray_{b}_{t}.png'
            #flow_gray_pil.save(filename)

    return noise_image
