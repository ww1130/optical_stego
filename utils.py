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
import torch.multiprocessing as mp
# from torch.fft import fft, ifft, fftshift, ifftshift
#from scipy.fft import dct, idct
# from torchdct import dct, idct
# from scipy.fftpack import dctn,idctn
import torch_dct as dct

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

def sobel_filter(x):
    """
    计算X和Y方向的梯度
    :param x: 输入的灰度张量 (b, 1, h, w)
    :return: X和Y方向的梯度张量 (b, 2, h, w)
    """
    sobel_kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device).view(1, 1, 3, 3)
    sobel_kernel_y = sobel_kernel_x.transpose(2, 3)
    
    grad_x = F.conv2d(x, sobel_kernel_x, padding=1)
    grad_y = F.conv2d(x, sobel_kernel_y, padding=1)
    
    return torch.cat((grad_x, grad_y), dim=1)

def magnitude(grads):
    """
    计算梯度幅值
    :param grads: 梯度张量 (b, 2, h, w)
    :return: 梯度幅值张量 (b, 1, h, w)
    """
    return torch.sqrt(torch.sum(grads ** 2, dim=1, keepdim=True))

def dct_2d(x):
    x = dct.dct_2d(x)
    return x

def idct_2d(x):
    x = dct.idct_2d(x)
    return x

def calculate_complexity(image_block):
    """
    Calculate the complexity of an image block based on its gradient
    :param image_block: Tensor of shape (c, height, width)
    :return: Complexity score
    """
    grad_x = torch.abs(image_block[:, :, 1:, :] - image_block[:, :, :-1, :])
    grad_y = torch.abs(image_block[:, :, :, 1:] - image_block[:, :, :, :-1])
    complexity = torch.mean(grad_x) + torch.mean(grad_y)
    return complexity
def zigzag_indices(height, width):
    indices = []
    for i in range(height + width - 1):
        if i < height:
            row_start, col_start = i, 0
        else:
            row_start, col_start = height - 1, i - height + 1
        while row_start >= 0 and col_start < width:
            indices.append((row_start, col_start))
            row_start -= 1
            col_start += 1
    return indices

def adaptive_block_division(flow_tensor, img_tensor, threshold_small=1000, threshold_large=3000, min_block_size=16):
    b, _, c, h, w = img_tensor.size()
    noisytensor = img_tensor.clone()

    # 计算梯度
    flow_tensor = flow_tensor.squeeze(1)  # (b, 1, h, w)
    #grads = sobel_filter(flow_tensor)
    #grad_magnitude = magnitude(grads)

    def process_block(x, y, width, height, batch_idx):
        stack = [(x, y, width, height)]
        while stack:
            x, y, width, height = stack.pop()
            if width >= 32:
                current_threshold = threshold_small
            else:
                current_threshold = threshold_large
            
            #block_grad = grad_magnitude[batch_idx, :, y:y + height, x:x + width]
            block_flow = flow_tensor[batch_idx, :, y:y + height, x:x + width]
            
            #if torch.max(block_grad) > current_threshold and width > min_block_size and height > min_block_size:
            if torch.var(block_flow) > current_threshold and width > min_block_size and height > min_block_size:
                half_width = width // 2
                half_height = height // 2
                
                stack.append((x, y, half_width, half_height))
                stack.append((x + half_width, y, half_width, half_height))
                stack.append((x, y + half_height, half_width, half_height))
                stack.append((x + half_width, y + half_height, half_width, half_height))
            else:
                block_brightness = torch.mean(flow_tensor[batch_idx, :, y:y + height, x:x + width])
                
                block = img_tensor[batch_idx, :, :, y:y + height, x:x + width]
                block_complexity = calculate_complexity(block)
                #zero_ratio = 0.7 + 0.2 * (block_brightness / 255.0)
                zero_ratio = 0.7 + 0.1 * (block_brightness / 255.0) + 0.2 * (block_complexity / 255.0)
                #zero_ratio2 = 0.7 + 0.2 * (block_brightness / 255.0) + 0.2 * (block_complexity / 255.0)
                zero_ratio = max(0.7, min(zero_ratio, 0.9))  # 限制在0.7到0.95之间

                
                block_dct = torch.stack([dct_2d(block[0, i]) for i in range(block.size(1))], dim=0).unsqueeze(0)


                zigzag_idx = zigzag_indices(height, width)
                block_dct_flattened = block_dct.view(c, -1)
                block_dct_zigzag = block_dct_flattened[:, [idx[0] * width + idx[1] for idx in zigzag_idx]]
                num_coeffs = block_dct_zigzag.size(1)  
                num_zeros = int(num_coeffs * zero_ratio)
                start=num_coeffs-num_zeros
                block_dct_zigzag[:, start:] = 0
                block_dct_flattened[:, [idx[0] * width + idx[1] for idx in zigzag_idx]] = block_dct_zigzag
                block_dct_zero = block_dct_flattened.view(c, height, width).unsqueeze(0)

                #block_idct = idct_2d(block_dct.view(c, height, width)).view(1, c, height, width)
                block_idct = torch.stack([idct_2d(block_dct_zero[0,i]) for i in range(block_dct.size(1))], dim=0)
                noisytensor[batch_idx, :, :, y:y + height, x:x + width] = block_idct
                #mse=torch.mean((noisytensor[batch_idx, :, :, y:y + height, x:x + width] - img_tensor[batch_idx, :, :, y:y + height, x:x + width]) ** 2)
                #pass
                #cv2.rectangle(out_flow_np, (x, y), (x + width, y + height), (0, 0, 128), 1)

    for batch_idx in range(b):
        # out_flow=flow_tensor[batch_idx,:,:,:].clone()
        # out_flow = out_flow.squeeze(0).squeeze(0)
        # out_flow_np = out_flow.cpu().numpy()
        # output_path = '/mnt/workspace/optical_stego/output_grid_image.png'
        for y in range(0, h, 64):
            for x in range(0, w, 64):
                width = min(64, w - x)
                height = min(64, h - y)
                process_block(x, y, width, height, batch_idx)
        # cv2.imwrite(output_path, out_flow_np)
        # print(f"结果已保存到: {output_path}")
        # pass

    return noisytensor

    
def print_decoder_gradients(decoder):
    for name, param in decoder.named_parameters():
        if param.requires_grad:
            gradient_status = "有梯度" if param.grad is not None else "无梯度"
            print(f"Layer: {name} - {gradient_status}")

def log_to_file(log_message, log_file="train.log"):
  """将日志消息追加到指定文件中。

  Args:
    log_message: 要写入日志文件的字符串。
    log_file: 日志文件的名称，默认为 "train.log"。
  """
  with open(log_file, "a") as f:
    f.write(log_message + "\n")

def MSE(host,stego):
    mse = torch.mean((host - stego) ** 2)
    return mse

def ACC(secret,rs):
    b=secret.shape[0]
    rs_sig=torch.sigmoid(rs)
    num05 = (rs_sig > 0.5).float()
    correct_bits = (num05 == secret).float()
    correct_count = correct_bits.sum().item()
    acc = correct_count / (256*448*b)
    return acc


def save_image(image, path):
    image_np = image.detach().cpu().numpy()  # 如果在GPU上，先转到CPU
    img = Image.fromarray(image_np.astype(np.uint8))  # 直接从0-255范围的数据创建图像
    img.save(path)


