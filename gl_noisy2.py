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
from dataset import Vimeo90kDatasettxtNoisy


# 加载数据集
bs=64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs=2
print_every_batch=bs
generate_secret_every_batch=bs
#dataset = Vimeo90kDataset('data')
dataset = Vimeo90kDatasettxtNoisy(root_dir='/home/admin/workspace/vimeo_triplet')
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

save_dir = './tripdata_model_imporedEn_DenseDe_10/'
encoder_save_path = os.path.join(save_dir, 'encoder_model.pth')
decoder_save_path = os.path.join(save_dir, 'decoder_model.pth')
# 训练过程
encoder= models.DenseEncoderNoisy().cuda()
decoder= models.DenseDecoderNoisy().cuda()

encoder.load_state_dict(torch.load(encoder_save_path))
decoder.load_state_dict(torch.load(decoder_save_path))

# Freeze all parameters in the encoder to make it non-trainable
for param in encoder.parameters():
    param.requires_grad = False

# Only include decoder parameters in the optimizer
optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

# optimizer = torch.optim.Adam(
#     list(encoder.parameters()) + list(decoder.parameters()),
#     lr=1e-4
# )


criterion = models.StegoLoss().cuda()

# encoder_mse_threshold_low = 5  # 当 MSE 小于此值时，loss=fnbit
# encoder_mse_threshold_high = 6 # 当 MSE 大于此值时，loss=mse+fnbit

loss_fn = nn.BCEWithLogitsLoss().cuda()

dwt_module = utils.DWT().cuda()
iwt_module = utils.IWT().cuda()


for epoch in range(epochs):
    running_loss = 0.0
    running_mse = 0.0
    running_mse_noisy=0.0
    running_fnbit = 0.0
    running_acc = 0.0
    batch_count = 0  # 用于计数当前epoch中的batch数
    
    # secret = torch.randint(0, 2, (bs, 6, 256, 448, 1), device=device).float()
    # secret = secret.permute(0, 1, 4, 2, 3)
    # secret_dwt_tensor=dwt_module(secret)
    
    #for host, flow_gray in dataloader:
    for batch_idx, (host,flow_gray) in enumerate(dataloader):
        optimizer.zero_grad()
        #secret = torch.randint(0, 2, flow_gray.shape).float()
        b,t,_,_,_=host.shape
        #防止最后一个批次的数据不够bs，导致报错
        if (batch_idx % generate_secret_every_batch == 0 or b != bs):
            #bs_last = host.size(0)  # 获取当前batch的大小
            secret = torch.randint(0, 2, (b, t, 256, 448, 1), device=device).float()
            secret = secret.permute(0, 1, 4, 2, 3)
            secret_dwt_tensor = dwt_module(secret)

        host = host.permute(0, 1, 4, 2, 3)
        flow_gray=flow_gray.permute(0, 1, 4, 2, 3)
        host_dwt_tensor = dwt_module(host)
        # secret_dwt_tensor = convert_to_tensor(secret_dwt)

        stego_dwt_tensor = encoder(host_dwt_tensor, secret_dwt_tensor)
       

        stego_image = iwt_module(stego_dwt_tensor)
        

        # 检查 stego_image 的形状
        # print(f"Shape of stego_image: {stego_image.shape}")
        #
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
        # #
        # stego_image_save = iwt_module(host_dwt_tensor).detach().squeeze(0).cpu().numpy()  # 去除大小为1的维度，并转为 numpy
        # # 循环保存每张stego
        # for i in range(stego_image_save.shape[0]):
        #     # 转换到 PIL Image 所需的形状 (H, W, C)
        #     # img = Image.fromarray((stego_image[i].transpose(1, 2, 0) * 255).astype('uint8'))  # 假设数据在0-1之间，需要转换到0-255
        #     img = Image.fromarray(stego_image_save[i].transpose(1, 2, 0).astype('uint8'))
        #     # 创建文件名
        #     filename = f"host{i + 1}.png"
        #     # 保存图片
        #     img.save(os.path.join(os.getcwd(), filename))
        #     print(f"Saved {filename}")

        # 检查 stego_image 的新形状
        # print(f"New shape of stego_image after adjustment: {stego_image.shape}")
        # 对每一个batch里每一张图片添加噪声
        # noisy_stego_image=[]
        # for b in range(len(stego_image)):
        #     noisy_stego_image_t=[]
        #     for t in range(len(stego_image[b])):
        #         stego_image_np = stego_image[b, t].permute(1, 2, 0).detach().numpy()  # (C, H, W) -> (H, W, C)
        #         # 噪声叠加
        #         noisy_stego = add_noise(stego_image_np, flow_gray.numpy())
        #         noisy_stego_image_t.append(noisy_stego)
        #
        #     noisy_stego_image.append(noisy_stego_image_t)
        #variance_threshold = 0.01

        noisy_stego_image = utils.add_noise_based_on_mean(stego_image, flow_gray,save_blocks=False)
        noisy_stego_image_tensor=dwt_module(noisy_stego_image)

        #保存添加添加噪声后的
        # noisy_stego_image_save=noisy_stego_image.detach().squeeze(0).cpu().numpy()
        # for t, noisy_stego_image_t in enumerate(noisy_stego_image_save):
        #     #for t, noisy_stego in enumerate(noisy_stego_image_t):
        #     img = Image.fromarray(noisy_stego_image_t.transpose(1, 2, 0).astype('uint8'))
        # #创建文件名
        #     filename = f"noisy{t+1}.png"
        #     # 保存图片
        #     img.save(os.path.join(os.getcwd(), filename))
        #     print(f"Saved {filename}")


        #rs = decoder(stego_image)
        rs_dwt = decoder(noisy_stego_image_tensor)
        rs=iwt_module(rs_dwt)


        rs_sig=torch.sigmoid(rs)


        fnbit = loss_fn(rs, secret)
        mse = criterion(host, stego_image)#, rs_binary, secret)
        mse_noisy = criterion(stego_image, noisy_stego_image)
        num05 = (rs_sig > 0.5).float()
        correct_bits = (num05 == secret).float()
        correct_count = correct_bits.sum().item()
        acc = correct_count / (256*448*b)



        # 
            
        loss =  fnbit  # 只关注提取质量

       

        loss.backward()
        optimizer.step()


        running_loss += loss.item()
        running_mse += mse.item()
        running_mse_noisy += mse_noisy.item()
        running_fnbit += fnbit.item()
        running_acc += acc
        batch_count += 1

        # 每32个batch打印一次
        if (batch_idx + 1) % print_every_batch == 0:
            avg_loss = running_loss / batch_count
            avg_mse = running_mse / batch_count
            avg_mse_noisy = running_mse_noisy / batch_count
            avg_fnbit = running_fnbit / batch_count
            avg_acc = running_acc / batch_count
            
            print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, MSE_NOISY: {avg_mse_noisy:.4f},FNBIT: {avg_fnbit:.4f}, Acc: {avg_acc:.4f}')
            #print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, FNBIT: {avg_fnbit:.4f}, Acc: {avg_acc:.4f}')
            # 重置累积值
            running_loss = 0.0
            running_mse = 0.0
            running_mse_noisy = 0.0
            running_fnbit = 0.0
            running_acc = 0.0
            batch_count = 0

            torch.cuda.empty_cache()

       
        

save_dir = './tripdata_model_imporedEn_DenseDe_10/'
os.makedirs(save_dir, exist_ok=True)
encoder_save_path = os.path.join(save_dir, 'encoder_model.pth')
decoder_save_path = os.path.join(save_dir, 'decoder_model.pth')


torch.save(encoder.state_dict(), encoder_save_path)
torch.save(decoder.state_dict(), decoder_save_path)

print(f'Model saved ')
