import os
import numpy as np
import cv2
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import models

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
        
        # #加载光流灰度图
        # flow_gray_path = os.path.join(sequence_path, f'flow_im1_im2_gray.png') 
        # if os.path.exists(flow_gray_path):
        #     flow_gray = cv2.imread(flow_gray_path, cv2.IMREAD_GRAYSCALE)

        #     if flow_gray is None:
        #         raise FileNotFoundError(f"Flow gray image {flow_gray_path} cannot be read.")
        # else:
        #     raise FileNotFoundError(f"Flow gray image {flow_gray_path} does not exist.")
        
        # #转换为 tensor 格式
        # images = np.stack(images, axis=0)  # (7, H, W, C)
        # flow_gray = np.expand_dims(flow_gray, axis=0)  # (1, H, W)

        # return torch.FloatTensor(images), torch.FloatTensor(flow_gray)
        
        #my fixs
        flow_grays=[]
        for i in range(1,7):
            flow_gray_path = os.path.join(sequence_path, f'flow_im{i}_im{i+1}_gray.png') 
            if os.path.exists(flow_gray_path):
                flow_gray = cv2.imread(flow_gray_path, cv2.IMREAD_GRAYSCALE)

                if flow_gray is not None:
                    flow_grays.append(flow_gray)
            else:
                raise FileNotFoundError(f"Flow gray image {flow_gray_path} does not exist.")
        
        # 转换为 tensor 格式
        images = np.stack(images, axis=0)  # (7, H, W, C)

        flow_grays = np.stack(flow_grays, axis=0)
        flow_grays = np.expand_dims(flow_grays, axis=-1)  # (6, H, W, 1)
        
        return torch.FloatTensor(images), torch.FloatTensor(flow_grays)


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
# class Encoder(nn.Module):
#     def __init__(self, in_channels=3, t_size=6, num_bands=4):
#         super(Encoder, self).__init__()
#         self.channel_attention = ChannelAttention()
#         self.spatial_attention = SpatialAttention()
#         #self.temporal_attention = TemporalAttention(t_size)
#         #self.band_attention = BandAttention(in_channels, num_bands)
#
#         # 添加特征提取的卷积层
#         self.conv1 = nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1)
#
#     def forward(self, host_dwt, secret_dwt):
#         # 提取注意力信息
#         host_dwt_sliced = host_dwt[:, 1: :, :, :, :, :]
#         channel_attention = self.channel_attention(host_dwt_sliced)  # 通道注意力
#         spatial_attention = self.spatial_attention(host_dwt_sliced)  # 空间注意力
#         #temporal_attention = self.temporal_attention(host_dwt)  # 时间注意力
#         #band_attention = self.band_attention(host_dwt)  # 频带注意力
#
#         # 使用 host_dwt 的注意力信息调整 secret_dwt
#         adjusted_secret_dwt = secret_dwt * channel_attention * spatial_attention
#         #* temporal_attention * band_attention
#
#         # 进行卷积操作
#         features = self.conv1(adjusted_secret_dwt)  # 第一个卷积层
#         features = nn.ReLU()(features)  # 激活函数
#         features = self.conv2(features)  # 第二个卷积层
#
#         # 残差连接
#         output = features + host_dwt  # 将卷积特征与 host_dwt 相加
#
#         return output  # 返回调整后的 secret_dwt

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

class StegoLoss(nn.Module):
    def __init__(self):
        super(StegoLoss, self).__init__()

    def forward(self, original, stego, extracted_secret, original_secret):
        psnr_loss = torch.mean((original - stego) ** 2) 

        bitwise_loss = torch.mean((extracted_secret - original_secret) ** 2)
        return psnr_loss + bitwise_loss


# 加载数据集
dataset = Vimeo90kDataset('data')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# 训练过程
#encoder = Encoder()  #通道数设为 1
encoder= models.Encoder()
# encoderlh=models.Encoder()
# encoderhl=models.Encoder()
# encoderhh=models.Encoder()
optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
# optimizerlh = optim.Adam(encoderlh.parameters(), lr=1e-3)
# optimizerhl = optim.Adam(encoderhl.parameters(), lr=1e-3)
# optimizerhh = optim.Adam(encoderhh.parameters(), lr=1e-3)
criterion = StegoLoss()

for epoch in range(10):
    for host, flow_gray in dataloader:
        secret = torch.randint(0, 2, flow_gray.shape).float()
        #secret = torch.randint(0, 2, host.shape).float() 

        # DWT 变换
        host = host.permute(0, 1, 4, 2, 3)
        secret=secret.permute(0, 1, 4, 2, 3)
        flow_gray=flow_gray.permute(0, 1, 4, 2, 3)

        host_dwt = dwt_transform(host)
        secret_dwt = dwt_transform(secret) #(B,T,C,4,H/2,W/2)


        #全部转换为tensor
        host_dwt_tensor = convert_to_tensor(host_dwt)
        secret_dwt_tensor = convert_to_tensor(secret_dwt)
        #每个频带拆开(b,t,3,h/2,w/2)
        hll, hlh, hhl, hhh = torch.split(host_dwt_tensor, 1, dim=3)
        hll = hll.squeeze(3)  # 移除第四个维度
        hlh = hlh.squeeze(3)
        hhl = hhl.squeeze(3)
        hhh = hhh.squeeze(3)

        sll, slh, shl, shh = torch.split(secret_dwt_tensor, 1, dim=3)
        sll = sll.squeeze(3)  # 移除第四个维度
        slh = slh.squeeze(3)
        shl = shl.squeeze(3)
        shh = shh.squeeze(3)


        stego_ll,stego_lh,stego_hl,stego_hh = encoder(hll,hlh ,hhl,hhh,sll,slh,shl,shh)
        # 将stego的各个频带拼在一起形成(B,T,C,4,H/2,W/2)的tensor
        stego_ll = stego_ll.unsqueeze(3)
        stego_lh = stego_lh.unsqueeze(3)
        stego_hl = stego_hl.unsqueeze(3)
        stego_hh = stego_hh.unsqueeze(3)
        # stego_dwt = torch.cat((hll, hlh, hhl, hhh), dim=3)
        stego_dwt=torch.cat((stego_ll, stego_lh, stego_hl, stego_hh), dim=3)
        # 拆分成列表，共计有
        #processor = StegoTensorProcessor(stego_dwt)
        result_list = StegoTensorProcessor(stego_dwt).process()

        stego_image = dwt_inverse(result_list)

        # 检查 stego_image 的形状
        # print(f"Shape of stego_image: {stego_image.shape}")
        #
        stego_image_save = stego_image.squeeze(0).squeeze(2).cpu().numpy()  # 去除大小为1的维度，并转为 numpy
        #循环保存每张图片
        for i in range(stego_image_save.shape[0]):
            # 转换到 PIL Image 所需的形状 (H, W, C)
            #img = Image.fromarray((stego_image[i].transpose(1, 2, 0) * 255).astype('uint8'))  # 假设数据在0-1之间，需要转换到0-255
            img = Image.fromarray(stego_image_save[i].transpose(1, 2, 0).astype('uint8'))
            # 创建文件名
            filename = f"stego{i+1}.png"
            # 保存图片
            img.save(os.path.join(os.getcwd(), filename))
            print(f"Saved {filename}")
        

        # 将 6D 张量调整为 5D 张量
        if stego_image.dim() == 6:
            # 合并维度
            stego_image = stego_image.view(stego_image.size(0), stego_image.size(1), stego_image.size(2), stego_image.size(3) * stego_image.size(4), stego_image.size(-1))

        # 确保 stego_image 的通道数为 3，与原始 host 图像一致
        # if stego_image.shape[-1] != 3:
           
        #     stego_image = stego_image[..., :3]

        # 检查 stego_image 的新形状
        # print(f"New shape of stego_image after adjustment: {stego_image.shape}")
        # 对没一个batch里每一张图片添加噪声
        noisy_stego_image=[]
        for b in range(len(stego_image)):
            noisy_stego_image_t=[]
            for t in range(len(stego_image[b])):
                stego_image_np = stego_image[b, t].permute(1, 2, 0).detach().numpy()  # (C, H, W) -> (H, W, C)
                # 噪声叠加
                noisy_stego = add_noise(stego_image_np, flow_gray.numpy())
                noisy_stego_image_t.append(noisy_stego)

            noisy_stego_image.append(noisy_stego_image_t)



        for b, noisy_stego_image_t in enumerate(noisy_stego_image):
            for t, noisy_stego in enumerate(noisy_stego_image_t):
                img = Image.fromarray(noisy_stego.astype('uint8'))
        #创建文件名
                filename = f"noisy{t+1}.png"
            # 保存图片
                img.save(os.path.join(os.getcwd(), filename))
                print(f"Saved {filename}")


        # # 噪声叠加
        # noisy_stego_image = add_noise(stego_image_np, flow_gray.numpy())

        #noisy_stego_image_tensor = torch.tensor(noisy_stego_image, dtype=torch.float32, requires_grad=True).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (H, W, C) -> (B=1, T=1, C, H, W)
        noisy_stego_image_tensor = torch.tensor(noisy_stego_image, dtype=torch.float32, requires_grad=True)
        noisy_stego_image_tensor = noisy_stego_image_tensor.permute(0, 1, 4, 2, 3)

        # 计算损失并更新
        # optimizerll.zero_grad()
        # optimizerlh.zero_grad()
        # optimizerhl.zero_grad()
        # optimizerhh.zero_grad()

        optimizer.zero_grad()


        # TOO:写decoder提取secret，decoder的输入是  stego_ll,stego_lh,stego_hl,stego_hh，输出是  hll,hlh,hhl,hhh  ,  sll,slh,shl,shh
        # hll,hlh ,hhl,hhh转换为extract_host     sll,slh,shl,shh转换为extract_secret

        # 维度匹配
        #loss = criterion(host, noisy_stego_image_tensor, secret, secret)  # 使用原始 secret 与提取的 secret 做比较
        loss = criterion(host, noisy_stego_image_tensor, extract_secret, secret)
        loss.backward()
        optimizer.step()



        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        

save_dir = './saved_models/'
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, 'final_model_test.pth')
torch.save(encoder.state_dict(), model_save_path)
print(f'Model saved at {model_save_path}')