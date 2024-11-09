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

# 加载数据集
bs=4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs=10
print_every_batch=32
generate_secret_every_batch=32
#dataset = Vimeo90kDataset('data')
dataset = Vimeo90kDatasettxt(root_dir='/home/admin/workspace/vimeo_septuplet')
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
# 训练过程
encoder= models.DenseEncoder().cuda()
decoder= models.DenseDecoder().cuda()
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=1e-4
)
criterion = models.StegoLoss().cuda()
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4096'

encoder_mse_threshold_low = 5  # 当 MSE 小于此值时，loss=fnbit
encoder_mse_threshold_high = 6 # 当 MSE 大于此值时，loss=mse+fnbit

loss_fn = nn.BCEWithLogitsLoss().cuda()

dwt_module = utils.DWT().cuda()
iwt_module = utils.IWT().cuda()


for epoch in range(epochs):
    running_loss = 0.0
    running_mse = 0.0
    running_fnbit = 0.0
    running_acc = 0.0
    batch_count = 0  # 用于计数当前epoch中的batch数
    
    secret = torch.randint(0, 2, (bs, 6, 256, 448, 1), device=device).float()
    secret = secret.permute(0, 1, 4, 2, 3)
    secret_dwt_tensor=dwt_module(secret)
    
    #for host, flow_gray in dataloader:
    for batch_idx, (host) in enumerate(dataloader):
        optimizer.zero_grad()
        #secret = torch.randint(0, 2, flow_gray.shape).float()
        b,_,_,_,_=host.shape
        # if (batch_idx % generate_secret_every_batch == 0):
        #     bs = host.size(0)  # 获取当前batch的大小
        #     secret = torch.randint(0, 2, (bs, 6, 256, 448, 1), device=device).float()
        #     secret = secret.permute(0, 1, 4, 2, 3)
        #     secret_dwt_tensor = dwt_module(secret)

        secret_dwt_tensor = secret_dwt_tensor[:b,:,:,:,:,:]
        secret=secret[:b,:,:,:,:]
        #print(flow_gray.shape)
        #secret = torch.randint(0, 2, host.shape).float() 

        host = host.permute(0, 1, 4, 2, 3)
        #host01=host[:,1:,:,:,:]
        # secret=secret.permute(0, 1, 4, 2, 3)
        #flow_gray=flow_gray.permute(0, 1, 4, 2, 3)

        #host_dwt = utils.dwt_transform(host)
        # secret_dwt = utils.dwt_transform(secret) #(B,T,C,4,H/2,W/2)

        #全部转换为tensor
        #host_dwt_tensor = convert_to_tensor(host_dwt)
        host_dwt_tensor = dwt_module(host)
        # secret_dwt_tensor = convert_to_tensor(secret_dwt)


        #host_dwt_tensor = host_dwt_tensor.requires_grad_(True)
        #secret_dwt_tensor = secret_dwt_tensor.requires_grad_(True)
        #optimizer.zero_grad()

        stego_dwt_tensor = encoder(host_dwt_tensor, secret_dwt_tensor)
        # 拆分成列表，共计有
        #processor = StegoTensorProcessor(stego_dwt_tensor)
        #result_list = StegoTensorProcessor(stego_dwt_tensor).process()

        #stego_image = utils.dwt_inverse(result_list)
        stego_image = iwt_module(stego_dwt_tensor)
        #stego_image = torch.tensor(stego_image, dtype=torch.float32, requires_grad=True)

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
        #
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


        # 将 6D 张量调整为 5D 张量
        # if stego_image.dim() == 6:
        #     # 合并维度
        #     stego_image = stego_image.view(stego_image.size(0), stego_image.size(1), stego_image.size(2), stego_image.size(3) * stego_image.size(4), stego_image.size(-1))

        # 确保 stego_image 的通道数为 3，与原始 host 图像一致
        # if stego_image.shape[-1] != 3:
           
        #     stego_image = stego_image[..., :3]

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

        #noisy_stego_image = utils.add_noise_based_on_variance(stego_image, flow_gray)

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

        #noisy_stego_image_tensor = torch.tensor(noisy_stego_image, dtype=torch.float32, requires_grad=True).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (H, W, C) -> (B=1, T=1, C, H, W)
        # noisy_stego_image_tensor = torch.tensor(noisy_stego_image, dtype=torch.float32, requires_grad=True)
        # noisy_stego_image_tensor = noisy_stego_image_tensor.permute(0, 1, 4, 2, 3)

        #rs_dwt = decoder(stego_dwt_tensor)
        rs = decoder(stego_image)

        # rh_list = StegoTensorProcessor(rh_dwt).process()
        #rs_list = StegoTensorProcessor(rs_dwt).process()
        #rs= utils.dwt_inverse(rs_list)
        #rs= iwt_module(rs_dwt)

        rs_sig=torch.sigmoid(rs)

        #rs_sig_flattened = rs_sig.flatten()
        #secret_flattened=secret.flatten()
        #rs_flattened=rs.flatten()

        #rs_binary = (rs_sig > 0.5).float()
        #correct_bits = (rs_binary == secret).float()
        #correct_count = correct_bits.sum().item()
        #precent=correct_count/22,020,096
        # 维度匹配
        #loss = criterion(host, noisy_stego_image_tensor, secret, secret)  # 使用原始 secret 与提取的 secret 做比较
        #loss = criterion(host, noisy_stego_image_tensor, rs_binary, secret)
        #psnr,acc = criterion(host, noisy_stego_image_tensor, rs_binary, secret)

        fnbit = loss_fn(rs, secret)
        mse = criterion(host, stego_image)#, rs_binary, secret)
        num05 = (rs_sig > 0.5).float()
        correct_bits = (num05 == secret).float()
        correct_count = correct_bits.sum().item()
        acc = correct_count / (256*448*6*b)

        del rs_sig, num05, correct_bits
        torch.cuda.empty_cache()

        # 根据MSE值动态调整损失函数
        if mse.item() < encoder_mse_threshold_low:
            # MSE已经足够低,转向优化信息提取的效果
            #print(f"Epoch {epoch+1}: MSE is good ({mse.item():.4f}), focusing on extraction quality")
            loss =  fnbit  # 只关注提取质量

        elif mse.item() > encoder_mse_threshold_high:
            # MSE过高,需要同时关注图像质量和提取效果
            #print(f"Epoch {epoch+1}: MSE is high ({mse.item():.4f}), balancing image quality and extraction")
            loss = mse +  fnbit  # 平衡图像质量和提取效果

        else:
            # MSE在可接受范围内,使用正常的损失计算
            loss =  fnbit  # 可以适当降低MSE的权重


        loss.backward()
        optimizer.step()

        # before_update = {name: param.data.clone() for name, param in decoder.named_parameters()}
        # optimizer.step()
        # after_update = {name: param.data for name, param in decoder.named_parameters()}
        #
        # for name in before_update:
        #     print(f"{name} updated: {not torch.equal(before_update[name], after_update[name])}")
        #
        # for name, param in decoder.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} gradient: {param.grad.abs().mean().item()}")
        #     else:
        #         print(f"{name} gradient: None")

        running_loss += loss.item()
        running_mse += mse
        running_fnbit += fnbit
        running_acc += acc
        batch_count += 1

        # 每32个batch打印一次
        if (batch_idx + 1) % print_every_batch == 0:
            avg_loss = running_loss / batch_count
            avg_mse = running_mse / batch_count
            avg_fnbit = running_fnbit / batch_count
            avg_acc = running_acc / batch_count
            
            print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, FNBIT: {avg_fnbit:.4f}, Acc: {avg_acc:.4f}')

            # 释放不必要的内存
            # del host, encoded_images, decoded_secret, secret, secret_dwt_tensor
            # torch.cuda.empty_cache()
            
            # 重置累积值
            running_loss = 0.0
            running_mse = 0.0
            running_fnbit = 0.0
            running_acc = 0.0
            batch_count = 0

        #print(f'Epoch {epoch + 1}, Loss: {loss.item()}, mse: {mse},fnbit:{fnbit},acc:{acc}')
        

save_dir = './saved_models_whole_data/'
os.makedirs(save_dir, exist_ok=True)
encoder_save_path = os.path.join(save_dir, 'encoder_model.pth')
decoder_save_path = os.path.join(save_dir, 'decoder_model.pth')


torch.save(encoder.state_dict(), encoder_save_path)
torch.save(decoder.state_dict(), decoder_save_path)

print(f'Model saved ')


# 可以设置动态权重
# def get_weights(mse_value):
#     if mse_value < encoder_mse_threshold_low:
#         return 0.0, 1.0  # (mse_weight, msebit_weight)
#     elif mse_value > encoder_mse_threshold_high:
#         return 1.0, 5.0
#     else:
#         # 可以根据MSE值动态调整权重
#         ratio = (mse_value - encoder_mse_threshold_low) / (encoder_mse_threshold_high - encoder_mse_threshold_low)
#         return ratio, 5.0
#
# # 在训练循环中使用
# mse_weight, msebit_weight = get_weights(mse.item())
# loss = mse_weight * mse + msebit_weight * msebit

# mse, msebit = criterion(host, stego_image, rs_binary, secret)
#
# # 动态冻结/解冻encoder并调整损失函数
# if mse.item() < encoder_mse_threshold_low and not encoder_frozen:
#     print(f"Epoch {epoch}: Encoder MSE dropped below {encoder_mse_threshold_low}, freezing encoder.")
#     for param in encoder.parameters():
#         param.requires_grad = False
#     encoder_frozen = True
#     # encoder冻结时,损失只考虑msebit
#     loss = msebit
#
# elif mse.item() > encoder_mse_threshold_high and encoder_frozen:
#     print(f"Epoch {epoch}: Encoder MSE exceeded {encoder_mse_threshold_high}, unfreezing encoder.")
#     for param in encoder.parameters():
#         param.requires_grad = True
#     encoder_frozen = False
#     # encoder解冻时,损失考虑mse和msebit的组合
#     loss = mse + 5 * msebit
#
# else:
#     # 正常情况下的损失计算
#     loss = mse + 5 * msebit if not encoder_frozen else msebit
#
# # 反向传播
# loss.backward()
# optimizer.step()
# optimizer.zero_grad()
#
# # 更新运行中的损失
# running_loss += loss.item()
# running_mse += mse.item()
# running_msebit += msebit.item()
#
# # 计算epoch平均损失
# epoch_loss = running_loss / len(train_loader)
# epoch_mse = running_mse / len(train_loader)
# epoch_msebit = running_msebit / len(train_loader)
#
# print(
#     f"Epoch {epoch}: Loss={epoch_loss:.4f}, MSE={epoch_mse:.4f}, MSEbit={epoch_msebit:.4f}, Encoder {'Frozen' if encoder_frozen else 'Active'}")