import torch
import os
from models import DenseEncoder, DenseDecoder,StegoLoss,StegoLosstest
from utils import dwt_transform, dwt_inverse,convert_to_tensor,StegoTensorProcessor
from dataset import Vimeo90kDataset
from torch.utils.data import DataLoader
import utils
from PIL import Image
import math
import numpy as np
import torch.nn as nn

# 定义损失函数
criterion = StegoLosstest()

# 加载数据集
dataset = Vimeo90kDataset('data')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 加载保存的模型
save_dir = './saved_models/'
encoder_save_path = os.path.join(save_dir, 'encoder_model.pth')
decoder_save_path = os.path.join(save_dir, 'decoder_model.pth')

encoder = DenseEncoder()
decoder = DenseDecoder()

# 加载模型权重
encoder.load_state_dict(torch.load(encoder_save_path))
decoder.load_state_dict(torch.load(decoder_save_path))

# 将模型设置为评估模式
encoder.eval()
decoder.eval()

dwt_module = utils.DWT().cuda()
iwt_module = utils.IWT().cuda()
loss_fn = nn.BCEWithLogitsLoss()
# 测试过程
with torch.no_grad():
    for host, flow_gray in dataloader:
        secret = torch.randint(0, 2, flow_gray.shape).float()
        # secret = torch.randint(0, 2, host.shape).float()

        # DWT 变换
        host = host.permute(0, 1, 4, 2, 3)
        secret = secret.permute(0, 1, 4, 2, 3)
        flow_gray = flow_gray.permute(0, 1, 4, 2, 3)

        secret_dwt_tensor = dwt_module(secret)
        host_dwt_tensor = dwt_module(host)
        #secret_rev=iwt_module(secret_dwt_tensor)
        stego_dwt_tensor = encoder(host_dwt_tensor, secret_dwt_tensor)

        stego_image = iwt_module(stego_dwt_tensor)

        # stego_image_save = stego_image.detach().squeeze(0).cpu().numpy()  # 去除大小为1的维度，并转为 numpy
        # #循环保存每张stego
        # for i in range(stego_image_save.shape[0]):
        #     # 转换到 PIL Image 所需的形状 (H, W, C)
        #     #img = Image.fromarray((stego_image[i].transpose(1, 2, 0) * 255).astype('uint8'))  # 假设数据在0-1之间，需要转换到0-255
        #     img = Image.fromarray(stego_image_save[i].transpose(1, 2, 0).astype('uint8'))
        #     # 创建文件名
        #     filename = f"stego{i+1}.png"
        #     # 保存图片
        #     img.save(os.path.join(os.getcwd(), filename))
        #     print(f"Saved {filename}")


        # 检查 stego_image 的新形状
        # print(f"New shape of stego_image after adjustment: {stego_image.shape}")
        # 对每一个batch里每一张图片添加噪声
        # noisy_stego_image = []
        # for b in range(len(stego_image)):
        #     noisy_stego_image_t = []
        #     for t in range(len(stego_image[b])):
        #         stego_image_np = stego_image[b, t].permute(1, 2, 0).detach().numpy()  # (C, H, W) -> (H, W, C)
        #         # 噪声叠加
        #         noisy_stego = utils.add_noise(stego_image_np, flow_gray.numpy())
        #         noisy_stego_image_t.append(noisy_stego)
        #
        #     noisy_stego_image.append(noisy_stego_image_t)

        # #保存添加添加噪声后的
        # for b, noisy_stego_image_t in enumerate(noisy_stego_image):
        #     for t, noisy_stego in enumerate(noisy_stego_image_t):
        #         img = Image.fromarray(noisy_stego.astype('uint8'))
        # #创建文件名
        #         filename = f"noisy{t+1}.png"
        #     # 保存图片
        #         img.save(os.path.join(os.getcwd(), filename))
        #         print(f"Saved {filename}")

        # # 噪声叠加
        # noisy_stego_image = add_noise(stego_image_np, flow_gray.numpy())

        # noisy_stego_image_tensor = torch.tensor(noisy_stego_image, dtype=torch.float32, requires_grad=True).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (H, W, C) -> (B=1, T=1, C, H, W)
        # noisy_stego_image_tensor = torch.tensor(noisy_stego_image, dtype=torch.float32, requires_grad=True)
        # noisy_stego_image_tensor = noisy_stego_image_tensor.permute(0, 1, 4, 2, 3)

        rs = decoder(stego_image)

        #rs = iwt_module(rs_dwt)
        rs_sig = torch.sigmoid(rs)
        msebit = loss_fn(rs, secret)
        #rs_binary = (rs_binary > 0.7).float()
        num05=(rs_sig > 0.5).float()

        # num_secret_than0=(secret>0).sum().item()
        # num_greater_than_0 = (rs_sig > 0).sum().item()
        # num_greater_than_03 = (rs_sig > 0.3).sum().item()
        # num_greater_than_05 = (rs_sig > 0.5).sum().item()
        # num_greater_than_07 = (rs_sig > 0.7).sum().item()
        one=(secret==1.0).float().sum().item()
        correct_bits = (num05 == secret).float()
        correct_count = correct_bits.sum().item()
        percent=correct_count/688128
        #msebit = torch.mean((rs_binary - secret) ** 2)
        # 维度匹配
        # loss = criterion(host, noisy_stego_image_tensor, secret, secret)  # 使用原始 secret 与提取的 secret 做比较
        #loss = criterion(host, noisy_stego_image_tensor, rs_binary, secret)
        mse,acc = criterion(host, stego_image, num05, secret)
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))
        # print(f'psnr: {psnr.item()} acc: {1-acc.item()}')
        print(f'psnr: {psnr} acc: { acc.item()}')

        # 可以在这里添加代码来保存或显示结果图像
        # ...

print('Testing complete.')