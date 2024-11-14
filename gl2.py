import os
import numpy as np
import cv2
#import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import models
import utils
from dataset import Vimeo90kDatasettxt

# 4. 融合与逆 DWT
def steganography(host_dwt, secret_dwt, lambda_factor=0.1):
    return host_dwt + lambda_factor * secret_dwt

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

def convert_to_tensor(item):
    if isinstance(item, torch.Tensor):
        return item
    elif isinstance(item, (list, tuple)):
        return torch.stack([convert_to_tensor(subitem) for subitem in item])
    else:
        return torch.tensor(item)


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

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数配置
bs = 4
epochs = 1
print_every_batch = 32
generate_secret_every_batch = 32
save_dir = './model_imporedEn_DenseDe_1001/'

# 数据集和数据加载器
dataset = Vimeo90kDatasettxt(root_dir='/home/admin/workspace/vimeo_septuplet')
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

# 模型和优化器
encoder = models.DenseEncoder().to(device)
decoder = models.DenseDecoder().to(device)

# 加载预训练模型
encoder_save_path = os.path.join(save_dir, 'encoder_model.pth')
decoder_save_path = os.path.join(save_dir, 'decoder_model.pth')

encoder.load_state_dict(torch.load(encoder_save_path))
decoder.load_state_dict(torch.load(decoder_save_path))

# 冻结 encoder 的参数
for param in encoder.parameters():
    param.requires_grad = False

# 优化器只包含 decoder 的参数
optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

# 损失函数
criterion = models.StegoLoss().to(device)
loss_fn = nn.BCEWithLogitsLoss().to(device)
dwt_module = utils.DWT().to(device)
iwt_module = utils.IWT().to(device)

# 阈值
encoder_mse_threshold_low = 5  # 当 MSE 小于此值时，loss=fnbit
encoder_mse_threshold_high = 6  # 当 MSE 大于此值时，loss=mse+fnbit

# 训练过程
for epoch in range(epochs):
    running_loss = 0.0
    running_mse = 0.0
    running_fnbit = 0.0
    running_acc = 0.0
    batch_count = 0  # 用于计数当前epoch中的batch数
    
    for batch_idx, (host) in enumerate(dataloader):
        optimizer.zero_grad()
        
        b, _, _, _, _ = host.shape
        if (batch_idx % generate_secret_every_batch == 0 or b != 32):
            secret = torch.randint(0, 2, (b, 6, 256, 448, 1), device=device).float()
            secret = secret.permute(0, 1, 4, 2, 3)
            secret_dwt_tensor = dwt_module(secret)

        host = host.permute(0, 1, 4, 2, 3)
        host_dwt_tensor = dwt_module(host)
        
        stego_dwt_tensor = encoder(host_dwt_tensor, secret_dwt_tensor)
        stego_image = iwt_module(stego_dwt_tensor)
        
        rs = decoder(stego_image)
        rs_sig = torch.sigmoid(rs)

        fnbit = loss_fn(rs, secret)
        mse = criterion(host, stego_image)
        num05 = (rs_sig > 0.5).float()
        correct_bits = (num05 == secret).float()
        correct_count = correct_bits.sum().item()
        acc = correct_count / (256 * 448 * 6 * b)

        # 根据MSE值动态调整损失函数
        
        # MSE在可接受范围内,使用正常的损失计算
        loss = fnbit  # 可以适当降低MSE的权重

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_mse += mse.item()
        running_fnbit += fnbit.item()
        running_acc += acc
        batch_count += 1

        # 每32个batch打印一次
        if (batch_idx + 1) % print_every_batch == 0:
            avg_loss = running_loss / batch_count
            avg_mse = running_mse / batch_count
            avg_fnbit = running_fnbit / batch_count
            avg_acc = running_acc / batch_count
            
            print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, FNBIT: {avg_fnbit:.4f}, Acc: {avg_acc:.4f}')

            # 重置累积值
            running_loss = 0.0
            running_mse = 0.0
            running_fnbit = 0.0
            running_acc = 0.0
            batch_count = 0

# 保存模型
os.makedirs(save_dir, exist_ok=True)
encoder_save_path = os.path.join(save_dir, 'encoder_model.pth')
decoder_save_path = os.path.join(save_dir, 'decoder_model2.pth')

torch.save(encoder.state_dict(), encoder_save_path)
torch.save(decoder.state_dict(), decoder_save_path)

print(f'Model saved to {save_dir}')
