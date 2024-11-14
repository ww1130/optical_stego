import os
import numpy as np
import cv2
#import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image,ImageDraw,ImageFont
import models
import torch.nn.functional as F

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


def add_noise_based_on_variance(stego_image, flow_gray):
    b, t, c, h, w = stego_image.shape
    
    # 确保flow_gray的shape正确
    assert flow_gray.shape == (b, t, 1, h, w), "Flow gray shape mismatch"
    
    noisy_stego_image = stego_image.clone()
    
    for batch in range(b):
        for time in range(t):
            # 对每个时间步进行处理
            current_flow = flow_gray[batch, time, 0]
            current_stego = stego_image[batch, time]
            
            # 自适应分块
            for block_size in [32, 16, 8]:
                # 计算需要的padding
                pad_h = (block_size - h % block_size) % block_size
                pad_w = (block_size - w % block_size) % block_size
                
                # 对flow和stego进行padding
                padded_flow = F.pad(current_flow, (0, pad_w, 0, pad_h))
                padded_stego = F.pad(current_stego, (0, pad_w, 0, pad_h))
                
                # 使用unfold进行分块
                flow_blocks = F.unfold(padded_flow.unsqueeze(0).unsqueeze(0), kernel_size=block_size, stride=block_size)
                flow_blocks = flow_blocks.view(1, block_size, block_size, -1).permute(0, 3, 1, 2)
                
                # 计算每个块的方差
                variances = torch.var(flow_blocks, dim=(2, 3))
                
                # 对stego image进行分块
                stego_blocks = F.unfold(padded_stego.unsqueeze(0), kernel_size=block_size, stride=block_size)
                stego_blocks = stego_blocks.view(c, block_size, block_size, -1).permute(0, 3, 1, 2)
                
                # 为每个通道生成噪声
                noise = torch.randn_like(stego_blocks) * variances.view(1, -1, 1, 1).sqrt()
                
                # 将噪声加到对应的stego image块上
                noisy_blocks = stego_blocks + noise
                
                # 将处理后的块放回原图
                noisy_blocks = noisy_blocks.permute(0, 2, 3, 1).reshape(c, -1, block_size*block_size)
                output = F.fold(noisy_blocks, output_size=(h+pad_h, w+pad_w), kernel_size=block_size, stride=block_size)
                
                # 移除padding
                output = output[:, :, :h, :w]
                
                # 更新noisy_stego_image
                noisy_stego_image[batch, time] = output.squeeze(0)
                
                # 更新current_stego为下一个block_size的输入
                current_stego = output.squeeze(0)
    
    return noisy_stego_image
def visualize_blocks_and_variances(flow_gray):
    b, t, _, h, w = flow_gray.shape
    
    for batch in range(b):
        for time in range(t):
            current_flow = flow_gray[batch, time, 0].cpu().numpy()
            
            plt.figure(figsize=(12, 12))
            plt.imshow(current_flow, cmap='gray')
            
            for block_size in [32, 16, 8]:
                blocks = F.unfold(flow_gray[batch, time], kernel_size=block_size, stride=block_size)
                blocks = blocks.view(1, block_size, block_size, -1).permute(0, 3, 1, 2)
                variances = torch.var(blocks, dim=(2, 3)).cpu().numpy()
                
                num_blocks_h = h // block_size
                num_blocks_w = w // block_size
                
                for i in range(num_blocks_h):
                    for j in range(num_blocks_w):
                        y = i * block_size
                        x = j * block_size
                        plt.gca().add_patch(plt.Rectangle((x, y), block_size, block_size, fill=False, edgecolor='r', linewidth=1))
                        plt.text(x + block_size//2, y + block_size//2, f'{variances[i*num_blocks_w + j]:.2f}', 
                                 color='r', ha='center', va='center')
            
            plt.title(f'Batch {batch}, Time {time}')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'flow_gray_blocks_batch{batch}_time{time}.png')
            plt.close()