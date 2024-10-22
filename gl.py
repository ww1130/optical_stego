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
import utils


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

# class StegoLoss(nn.Module):
#     def __init__(self):
#         super(StegoLoss, self).__init__()
#
#     def forward(self, originalHavefirstframe, stego, extracted_secret, original_secret):
#         original =originalHavefirstframe[:,1:,:,:,:]
#         psnr_loss = torch.mean((original - stego) ** 2)
#
#         # 计算比特准确率
#         correct_bits = (extracted_secret == original_secret).float().sum()
#         total_bits = original_secret.numel()  # numel() 返回张量的元素总数
#         bitwise_accuracy = correct_bits / total_bits
#         # 如果您想要将比特准确率也作为一个损失返回，可以如下操作：
#         bitwise_loss = 1 - bitwise_accuracy  # 这会将准确率转换为损失
#
#         # bitwise_loss = torch.mean((extracted_secret - original_secret) ** 2)
#
#         return psnr_loss , bitwise_loss


# 加载数据集
bs=32
epochs=10
dataset = Vimeo90kDataset('data')
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
# 训练过程
encoder= models.DenseEncoder()
decoder= models.DenseDecoder()
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=1e-4
)
criterion = models.StegoLoss()

encoder_mse_threshold_low = 1  # 当 MSE 小于此值时，冻结 encoder
encoder_mse_threshold_high = 2 # 当 MSE 大于此值时，重新开启 encoder 的优化
encoder_frozen = False

loss_fn = nn.BCEWithLogitsLoss()

dwt_module = utils.DWT().cuda()
iwt_module = utils.IWT().cuda()


for epoch in range(epochs):
    #print("test "  ,epoch)
    secret = torch.randint(0, 2, (bs,6,256,448,1)).float()
    secret = secret.permute(0, 1, 4, 2, 3)
    #secret_dwt = utils.dwt_transform(secret)  # (B,T,C,4,H/2,W/2)
    #secret_dwt_tensor = convert_to_tensor(secret_dwt)
    secret_dwt_tensor=dwt_module(secret)
    #secret_dwt_tensor=secret_dwt_tensor.unsqueeze(2)
    for host, flow_gray in dataloader:
        #secret = torch.randint(0, 2, flow_gray.shape).float()
        b,_,_,_,_=flow_gray.shape
        secret_dwt_tensor = secret_dwt_tensor[:b,:,:,:,:,:]
        secret=secret[:b,:,:,:,:]
        #print(flow_gray.shape)
        #secret = torch.randint(0, 2, host.shape).float() 

        host = host.permute(0, 1, 4, 2, 3)
        #host01=host[:,1:,:,:,:]
        # secret=secret.permute(0, 1, 4, 2, 3)
        flow_gray=flow_gray.permute(0, 1, 4, 2, 3)

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

        #optimizer.zero_grad()

        # TOO:写decoder提取secret，decoder的输入是  stego_ll,stego_lh,stego_hl,stego_hh，输出是  hll,hlh,hhl,hhh  ,  sll,slh,shl,shh
        # hll,hlh ,hhl,hhh转换为extract_host     sll,slh,shl,shh转换为extract_secret
        #rhll,rhlh,rhhl,rhhh,rsll,rslh,rshl,rshh=decoder(stego_ll,stego_lh,stego_hl,stego_hh)



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


        # 根据MSE值动态调整损失函数
        if mse.item() < encoder_mse_threshold_low:
            # MSE已经足够低,转向优化信息提取的效果
            print(f"Epoch {epoch+1}: MSE is good ({mse.item():.4f}), focusing on extraction quality")
            loss = 5 * fnbit  # 只关注提取质量

        elif mse.item() > encoder_mse_threshold_high:
            # MSE过高,需要同时关注图像质量和提取效果
            print(f"Epoch {epoch+1}: MSE is high ({mse.item():.4f}), balancing image quality and extraction")
            loss = mse + 5 * fnbit  # 平衡图像质量和提取效果

        else:
            # MSE在可接受范围内,使用正常的损失计算
            loss =   5 * fnbit  # 可以适当降低MSE的权重


        optimizer.zero_grad()
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

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}, mse: {mse},fnbit:{fnbit},acc:{acc}')
        

save_dir = './saved_models/'
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