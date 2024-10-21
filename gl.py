import os
import numpy as np
import cv2
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# 1. 数据集加载

class Vimeo90kDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 遍历两层目录
        self.sequences = []
        for subdir in os.listdir(root_dir):

            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for nested_subdir in os.listdir(subdir_path):
                    nested_subdir_path = os.path.join(subdir_path, nested_subdir)
                    if os.path.isdir(nested_subdir_path):
                        self.sequences.append(nested_subdir_path)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_path = self.sequences[idx]
        images = []

        # 加载7张连续的图片
        for i in range(1, 8):
            img_path = os.path.join(sequence_path, f'im{i}.png')
            if os.path.exists(img_path):
                img = cv2.imread(img_path)

                if img is not None:
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    raise FileNotFoundError(f"Image {img_path} cannot be read.")
            else:
                raise FileNotFoundError(f"Image {img_path} does not exist.")
        
        # 加载光流灰度图
        flow_gray_path = os.path.join(sequence_path, f'flow_im1_im2_gray.png') 
        if os.path.exists(flow_gray_path):
            flow_gray = cv2.imread(flow_gray_path, cv2.IMREAD_GRAYSCALE)

            if flow_gray is None:
                raise FileNotFoundError(f"Flow gray image {flow_gray_path} cannot be read.")
        else:
            raise FileNotFoundError(f"Flow gray image {flow_gray_path} does not exist.")
        
        # 转换为 tensor 格式
        images = np.stack(images, axis=0)  # (7, H, W, C)
        flow_gray = np.expand_dims(flow_gray, axis=0)  # (1, H, W)
        
        return torch.FloatTensor(images), torch.FloatTensor(flow_gray)

# 加载数据集
dataset = Vimeo90kDataset('data')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

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
                img_reconstructed = pywt.idwt2((ll.detach().numpy(), (lh.detach().numpy(), hl.detach().numpy(), hh.detach().numpy())), 'haar')
                time_imgs.append(torch.tensor(img_reconstructed))
            batch_imgs.append(torch.stack(time_imgs, dim=0))  # (C, H, W)
        reconstructed_images.append(torch.stack(batch_imgs, dim=0))  # (T, C, H, W)

    return torch.stack(reconstructed_images, dim=0)  # (B, T, C, H, W)

# 3. 注意力机制
class AttentionModule(nn.Module):
    def __init__(self, channels=1):
        super(AttentionModule, self).__init__()
        # 对于单通道输入，输入/输出通道数有效

        self.spatial_attention = nn.Conv2d(channels, 1, kernel_size=1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // 8), kernel_size=1),
            nn.ReLU(),

            nn.Conv2d(max(1, channels // 8), channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spatial_weight = self.spatial_attention(x)
        channel_weight = self.channel_attention(x)
        return x * spatial_weight * channel_weight

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
    return noisy_image


class StegoLoss(nn.Module):
    def __init__(self):
        super(StegoLoss, self).__init__()

    def forward(self, original, stego, extracted_secret, original_secret):
        psnr_loss = torch.mean((original - stego) ** 2) 

        bitwise_loss = torch.mean((extracted_secret - original_secret) ** 2)
        return psnr_loss + bitwise_loss

# 训练过程
encoder = AttentionModule(channels=1)  #通道数设为 1
optimizer = optim.Adam(encoder.parameters(), lr=1e-4)
criterion = StegoLoss()

for epoch in range(10):
    for host, flow_gray in dataloader:
        secret = torch.randint(0, 2, host.shape).float() 

        # DWT 变换
        host_dwt = dwt_transform(host)
        secret_dwt = dwt_transform(secret)

        # 注意力处理与融合
        for b in range(len(secret_dwt)):
            for t in range(len(secret_dwt[b])):
                for c in range(len(secret_dwt[b][t])):
                    ll, lh, hl, hh = secret_dwt[b][t][c]
                    # 使用 encoder 分别对 ll, lh, hl, hh 进行注意力调整
                    secret_dwt[b][t][c] = (

                        encoder(ll.unsqueeze(0).unsqueeze(0)).squeeze(0),  # (H/2, W/2)
                        encoder(lh.unsqueeze(0).unsqueeze(0)).squeeze(0),  # (H/2, W/2)
                        encoder(hl.unsqueeze(0).unsqueeze(0)).squeeze(0),  # (H/2, W/2)
                        encoder(hh.unsqueeze(0).unsqueeze(0)).squeeze(0)   # (H/2, W/2)
                    )

        # 融合过程
        stego_dwt = []
        for b in range(len(host_dwt)):
            batch_stego = []
            for t in range(len(host_dwt[b])):
                time_stego = []
                for c in range(len(host_dwt[b][t])):
                    host_coeffs = host_dwt[b][t][c]

                    secret_coeffs = secret_dwt[b][t][c]
                    # 逐频带融合
                    fused_coeffs = tuple(h + 0.1 * s for h, s in zip(host_coeffs, secret_coeffs))
                    time_stego.append(fused_coeffs)
                batch_stego.append(time_stego)
            stego_dwt.append(batch_stego)

        stego_image = dwt_inverse(stego_dwt)

        # 检查 stego_image 的形状
        # print(f"Shape of stego_image: {stego_image.shape}")

        # 将 6D 张量调整为 5D 张量
        if stego_image.dim() == 6:
            # 合并维度
            stego_image = stego_image.view(stego_image.size(0), stego_image.size(1), stego_image.size(2), stego_image.size(3) * stego_image.size(4), stego_image.size(-1))

        # 确保 stego_image 的通道数为 3，与原始 host 图像一致
        if stego_image.shape[-1] != 3:
           
            stego_image = stego_image[..., :3]

        # 检查 stego_image 的新形状
        # print(f"New shape of stego_image after adjustment: {stego_image.shape}")

        stego_image_np = stego_image[0, 0].permute(1, 2, 0).detach().numpy()  # (C, H, W) -> (H, W, C)

        # 噪声叠加
        noisy_stego_image = add_noise(stego_image_np, flow_gray.numpy())


        noisy_stego_image_tensor = torch.tensor(noisy_stego_image, dtype=torch.float32, requires_grad=True).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (H, W, C) -> (B=1, T=1, C, H, W)

        # 计算损失并更新
        optimizer.zero_grad()

        # 维度匹配
        loss = criterion(host, noisy_stego_image_tensor, secret, secret)  # 使用原始 secret 与提取的 secret 做比较
        loss.backward()
        optimizer.step()


        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')