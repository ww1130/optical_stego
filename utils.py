import os
import numpy as np
import cv2
#import pywt
import logging
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
import ffmpeg
import time
import subprocess

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
    #num05 = (rs > 0.5).float()
    correct_bits = (num05 == secret).float()
    correct_count = correct_bits.sum().item()
    acc = correct_count / (256*448*b)
    return acc


def rgb_to_yuv420(images):
    # 初始化用于保存所有时间步骤的结果的列表
    y_list, u_list, v_list = [], [], []

    for img in images:
        # 确保图像是uint8类型，并且值在0-255范围内
        if img.dtype != np.uint8:
            raise ValueError("输入图像应为uint8类型")
        
        # 如果图像是RGB格式，则转换为YUV (4:4:4)
        if len(img.shape) == 3 and img.shape[2] == 3:
            yuv_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        else:
            raise ValueError("图像不是有效的3通道RGB格式")

        # 分离Y、U、V通道
        y, u, v = cv2.split(yuv_image)

        # 下采样U和V通道至4:2:0格式
        u_downsampled = cv2.resize(u, (u.shape[1] // 2, u.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
        v_downsampled = cv2.resize(v, (v.shape[1] // 2, v.shape[0] // 2), interpolation=cv2.INTER_CUBIC)

        # 将结果添加到列表中
        y_list.append(torch.FloatTensor(y).to('cuda'))
        # u_list.append(torch.FloatTensor(u_downsampled).to('cuda'))
        # v_list.append(torch.FloatTensor(v_downsampled).to('cuda'))
        u_list.append(torch.FloatTensor(u).to('cuda'))
        v_list.append(torch.FloatTensor(v).to('cuda'))

    # 将所有时间步骤的结果堆叠在一起
    y_all = torch.stack(y_list, dim=0)  # 形状为 (t, h, w)
    u_all = torch.stack(u_list, dim=0)  # 形状为 (t, h//2, w//2)
    v_all = torch.stack(v_list, dim=0)  # 形状为 (t, h//2, w//2)

    return y_all, u_all, v_all


def save_yuv_sequence(y_all, u_all, v_all, output_dir='output', filename_prefix='test'):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # 清空目录中的所有文件
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子目录及其内容
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')   

    b, t, h, w = y_all.shape  # 获取批次大小、时间长度、高度和宽度
    _, _, uh, uw = u_all.shape  # 下采样的U和V的高度和宽度


    for batch_idx in range(b):
        filename = f"{filename_prefix}{batch_idx}.yuv"
        filepath = os.path.join(output_dir, filename)
        
        yuv_data = bytearray()
        for time_idx in range(t):
            y = y_all[batch_idx, time_idx].detach().cpu().numpy()
            u = u_all[batch_idx, time_idx].detach().cpu().numpy()
            v = v_all[batch_idx, time_idx].detach().cpu().numpy()
            yuv_data.extend(y.astype(np.uint8).tobytes())
            yuv_data.extend(u.astype(np.uint8).tobytes())
            yuv_data.extend(v.astype(np.uint8).tobytes())
        # 一次性写入所有帧的数据
        with open(filepath, 'wb') as f:
            f.write(yuv_data)

    # for batch_idx in range(b):
    #     filename = f"{filename_prefix}_{batch_idx:03d}.yuv"
    #     filepath = os.path.join(output_dir, filename)
    #     with open(filepath, 'wb') as f:
    #         # 将GPU上的张量转移到CPU并转换为NumPy数组
    #         for time_idx in range(t):
    #             y = y_all[batch_idx, time_idx].detach().cpu().numpy()
    #             u_downsampled = u_all[batch_idx, time_idx].detach().cpu().numpy()
    #             v_downsampled = v_all[batch_idx, time_idx].detach().cpu().numpy()

    #             # 写入YUV数据到文件
    #             # with open(filepath, 'wb') as f:
    #             f.write(y.astype(np.uint8).tobytes())
    #             f.write(u_downsampled.astype(np.uint8).tobytes())
    #             f.write(v_downsampled.astype(np.uint8).tobytes())
    

def merge_frames_into_sequence(y_all, u_all, v_all, stego_y_255, stego_uv_255):
    """
    将给定的帧合并成一个新的 YUV420 序列。
    
    参数：
    - y_all: 形状为 (b, t, 1, h, w) 的 Y 分量 tensor。
    - u_all: 形状为 (b, t, 1, h//2, w//2) 的 U 分量 tensor。
    - v_all: 形状为 (b, t, 1, h//2, w//2) 的 V 分量 tensor。
    - stego_y_255: 新的 Y 分量帧，形状为 (1, 1, h, w)，值域 [0, 255]。
    - stego_uv_255: 新的 U 和 V 分量帧，形状为 (1, 1, h//2, w//2)，值域 [0, 255]。
    
    返回：
    - 新的 YUV420 序列，包含原始的第一帧、插入的帧以及原始的最后一帧。
    """
    # 确保输入张量是 PyTorch 张量类型
    if not isinstance(stego_y_255, torch.Tensor):
        stego_y_255 = torch.tensor(stego_y_255)
    if not isinstance(stego_uv_255, torch.Tensor):
        stego_uv_255 = torch.tensor(stego_uv_255)

    # 提取第一帧和最后一帧
    first_frame_y = y_all[:, 0:1]
    last_frame_y = y_all[:, -1:]
    first_frame_u = u_all[:, 0:1]
    last_frame_v = v_all[:, -1:]

    # 准备新的帧（假定 stego_uv_255 已经被分割成两个独立的张量）
    stego_u_255, stego_v_255 = stego_uv_255.chunk(2, dim=1)

    # 将新帧的值从 [0, 255] 转换回 [0, 1] 或者适当的范围
    stego_y = stego_y_255.float() / 255.0
    stego_u = stego_u_255.float() / 255.0 - 0.5  # 假设原始偏移是 +128
    stego_v = stego_v_255.float() / 255.0 - 0.5  # 假设原始偏移是 +128

    # 合并 Y, U, V 分量
    new_y_sequence = torch.cat([first_frame_y, stego_y, last_frame_y], dim=1)
    new_u_sequence = torch.cat([first_frame_u, stego_u, last_frame_v], dim=1)
    new_v_sequence = torch.cat([first_frame_u, stego_v, last_frame_v], dim=1)

    return new_y_sequence, new_u_sequence, new_v_sequence



# def encode_yuv_to_hevc(yuv_file, video_path, width=448, height=256, fps=1, qp=32):
#     #ffmpeg.input(yuv_file, pix_fmt='yuv420p', video_size=f'{width}x{height}').output(video_path, vcodec='libx265', qp=qp).run(overwrite_output=True, quiet=True)
#     ffmpeg.input(yuv_file, pix_fmt='yuv444p', s=f'{width}x{height}').output(video_path, vcodec='libx265', qp=qp).run(quiet=True)

def encode_yuv_to_hevc(yuv_file, video_path, width=448, height=256, fps=1, qp=32):
    # 构建 FFmpeg 命令
    # ffmpeg_cmd = [
    #     'ffmpeg', '-y',                        # 覆盖输出文件
    #     '-pix_fmt', 'yuv444p',                 # 输入 YUV 文件的像素格式
    #     '-s', f'{width}x{height}',             # 输入视频的分辨率
    #     '-i', yuv_file,                        # 输入 YUV 文件
    #     '-c:v', 'libx265',                     # 视频编码器使用 libx265
    #     '-qp', str(qp),                        # 设置量化参数（QP）
    #     video_path                             # 输出文件路径
    # ]
    # ffmpeg_cmd= [
    #         'ffmpeg',
    #         '-y',
    #         '-f', 'rawvideo',
    #         '-pix_fmt', 'yuv444p',
    #         '-video_size', f'{width}x{height}',
    #         '-i', yuv_file,
    #         '-c:v', 'libx264',
    #         '-crf', str(32),
    #         video_path
    # ]
    ffmpeg_cmd= [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'yuv444p',
            '-video_size', f'{width}x{height}',
            '-i', yuv_file,
            '-c:v', 'libx264',
            '-qp', str(qp),
            video_path
    ]
    try:
        # 调用命令并捕获错误输出
        subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        # print("FFmpeg command:")
        # print(" ".join(ffmpeg_cmd))  # 将命令列表打印成字符串
        #print(f"Successfully encoded {yuv_file} to {video_path}")
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，打印完整的命令和错误信息
        print(f"Error occurred while encoding {yuv_file} to {video_path}")
        print("FFmpeg command:")
        print(" ".join(ffmpeg_cmd))  # 将命令列表打印成字符串
        print("Error output:")
        print(e.stderr.decode())  # 打印 FFmpeg 的标准错误输出

# def encode_yuv_to_hevc(yuv_file, video_path, width=448, height=256, fps=1, qp=32):
#     if not os.path.exists(yuv_file):
#         logging.error(f"YUV 文件 {yuv_file} 不存在")
#         raise FileNotFoundError(f"YUV 文件 {yuv_file} 不存在")

#     try:
#         # 构建 FFmpeg 命令
#         process = (
#             ffmpeg
#             .input(yuv_file, pix_fmt='yuv444p', s=f'{width}x{height}')
#             .output(video_path, vcodec='libx265', qp=qp)
#             .global_args('-loglevel', 'error')
#             .overwrite_output()
#         )

#         # 获取并打印 FFmpeg 命令行
#         cmd = ffmpeg.compile(process)
#         logging.info(f"Executing FFmpeg command: {' '.join(cmd)}")

#         # 运行 FFmpeg 命令
#         out, err = process.run(capture_stdout=True, capture_stderr=True)

#         logging.info(f"成功编码 {yuv_file} 至 {video_path}")
        
#     except ffmpeg._run.Error as e:
#         logging.error(f"FFmpeg Error while processing {yuv_file}:")
#         if e.stdout:
#             logging.error("stdout: %s", e.stdout.decode())
#         if e.stderr:
#             logging.error("stderr: %s", e.stderr.decode())
#         raise


def decode_hevc_to_yuv(video_path, yuv_output_pattern):
    ffmpeg.input(video_path).output(yuv_output_pattern, vcodec='rawvideo', pix_fmt='yuv444p').run(overwrite_output=True, quiet=True)

def extract_second_frame_from_yuv(yuv_data, width, height):
    # 计算YUV文件中每一帧的大小
    frame_size = width * height * 3 #// 2  # YUV420P: Y (w*h) + U (w/2*h/2) + V (w/2*h/2)
    
    # 跳过第一帧，读取第二帧的数据
    second_frame_start = frame_size
    second_frame_end = second_frame_start + frame_size
    
    second_frame_data = yuv_data[second_frame_start:second_frame_end]
    
    # 分离Y、U、V通道
    y_size = width * height
    u_size = v_size = y_size ##// 4
    
    y_data = np.frombuffer(second_frame_data[:y_size], dtype=np.uint8).reshape(height, width)
    u_data = np.frombuffer(second_frame_data[y_size:y_size + u_size], dtype=np.uint8).reshape(height , width )
    v_data = np.frombuffer(second_frame_data[y_size + u_size:second_frame_end], dtype=np.uint8).reshape(height , width )
    
    return y_data, u_data, v_data

def process_batch(b, outdir, width, height, fps=1, qp=27):
    final_y = []
    final_u = []
    final_v = []

    for i in range(b):
        # 定义路径
        yuv_file = os.path.join(outdir, f'test{i}.yuv')
        mp4_path = os.path.join(outdir, f'test{i}.mp4')
        decoded_yuv_path = os.path.join(outdir, f'decode{i}.yuv')

        
        # 编码 YUV 文件为 H.265 视频
        #time.sleep(0.5)
        encode_yuv_to_hevc(yuv_file, mp4_path, width, height, fps, qp)
        # 解码 H.265 视频回 YUV 文件
        decode_hevc_to_yuv(mp4_path, decoded_yuv_path)

        # 读取解码后的 YUV 文件内容
        with open(decoded_yuv_path, 'rb') as f:
            yuv_data = f.read()

        # 提取解码后的 YUV 文件中的第二帧
        y_data, u_data, v_data = extract_second_frame_from_yuv(yuv_data, width, height)
        
        # 将第二帧的数据添加到结果列表中
        final_y.append(y_data)
        final_u.append(u_data)
        final_v.append(v_data)


    # 将所有数据转换为 NumPy 数组
    final_y_np = np.stack(final_y, axis=0)  # shape: (b, height, width)
    final_u_np = np.stack(final_u, axis=0)  # shape: (b, height//2, width//2)
    final_v_np = np.stack(final_v, axis=0)  # shape: (b, height//2, width//2)

    # 将 NumPy 数组转换为 PyTorch 张量并移动到 CUDA 设备
    final_y_tensor = torch.from_numpy(final_y_np).to('cuda')
    final_u_tensor = torch.from_numpy(final_u_np).to('cuda')
    final_v_tensor = torch.from_numpy(final_v_np).to('cuda')

    # 返回张量
    return final_y_tensor, final_u_tensor, final_v_tensor


def save_image(image, path):
    image_np = image.detach().cpu().numpy()  # 如果在GPU上，先转到CPU
    img = Image.fromarray(image_np.astype(np.uint8))  # 直接从0-255范围的数据创建图像
    img.save(path)


# def save_yuv_sequence(images, output_path):
#     # 获取第一张图片的尺寸
#     height, width = images[0].shape[:2]
    
#     with open(output_path, 'wb') as f:
#         for img in images:
#             # 确保图像是uint8类型，并且值在0-255范围内
#             if img.dtype != np.uint8:
#                 raise ValueError("输入图像应为uint8类型")
            
#             # 如果图像是RGB格式，则转换为YUV (4:4:4)
#             if len(img.shape) == 3 and img.shape[2] == 3:
#                 yuv_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#             else:
#                 raise ValueError("图像不是有效的3通道RGB格式")

#             # 分离Y、U、V通道
#             y, u, v = cv2.split(yuv_image)

#             # 下采样U和V通道至4:2:0格式
#             u_downsampled = cv2.resize(u, (u.shape[1] // 2, u.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
#             v_downsampled = cv2.resize(v, (v.shape[1] // 2, v.shape[0] // 2), interpolation=cv2.INTER_CUBIC)

#             # 将每个通道的数据分别写入文件
#             # f.write(y.astype(np.uint8).tobytes())
#             # f.write(u.astype(np.uint8).tobytes())
#             # f.write(v.astype(np.uint8).tobytes())

#             f.write(y.astype(np.uint8).tobytes())
#             f.write(u_downsampled.astype(np.uint8).tobytes())
#             f.write(v_downsampled.astype(np.uint8).tobytes())


