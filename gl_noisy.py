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
import quant
from dataset import Vimeo90kDatasettxtNoisy


save_dir = './tripdata_model_imporedEn_secRes_DenseDe_nonoisy/'
encoder_save_path = os.path.join(save_dir, 'encoder_model.pth')
decoder_save_path = os.path.join(save_dir, 'decoder_model.pth')

log_file_path = 'train.log'
with open(log_file_path, 'w') as f:
    pass  # 这里什么都不做，只是清空文件

# 加载数据集
bs=64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs=2
print_every_batch=1
generate_secret_every_batch=1
dataset = Vimeo90kDatasettxtNoisy(root_dir='/home/admin/workspace/vimeo_triplet')
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True,drop_last=True)
quantization = quant.Quantization().cuda()
# 训练过程
encoder= models.DenseEncoderNoisy().cuda()
decoder= models.DenseDecoderNoisy().cuda()
# encoder.load_state_dict(torch.load(encoder_save_path))
# decoder.load_state_dict(torch.load(decoder_save_path))

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=1e-4)

encoder_mse_threshold_low = 6  # 当 MSE 小于此值时，loss=fnbit
encoder_mse_threshold_high = 8 # 当 MSE 大于此值时，loss=mse+fnbit

loss_fn = nn.BCEWithLogitsLoss().cuda()
mse_loss = nn.MSELoss().to(device)
dwt_module = utils.DWT().cuda()
iwt_module = utils.IWT().cuda()


for epoch in range(epochs):
    running_loss = 0.0
    running_mse = 0.0
    running_mse_noisy=0.0
    running_fnbit = 0.0
    running_acc = 0.0
    batch_count = 0  # 用于计数当前epoch中的batch数

    for batch_idx, (host,flow_gray,decoded) in enumerate(dataloader):
        b,t,_,_,_=host.shape
        #防止最后一个批次的数据不够bs，导致报错
        # if (batch_idx % generate_secret_every_batch == 0 or b != bs):
            #bs_last = host.size(0)  # 获取当前batch的大小
        secret = torch.randint(0, 2, (b, 1, 256, 448, 1), device=device).float()


        secret = secret.permute(0, 1, 4, 2, 3)
        host = host.permute(0, 1, 4, 2, 3)
        flow_gray=flow_gray.permute(0, 1, 4, 2, 3)
        decoded=decoded.permute(0, 1, 4, 2, 3)


        secret_dwt = dwt_module(secret)
        host_dwt = dwt_module(host)
        flow_gray_dwt = dwt_module(flow_gray)

        res_dwt = encoder(host_dwt, secret_dwt ,flow_gray_dwt)

        res = iwt_module(res_dwt)
        res=res.permute(0,2,1,3,4)
        res=models.framenorm(res)
        res=res.permute(0,2,1,3,4)

        stego=(host+res).clamp(-1,1)
        stego_dwt = dwt_module(stego)

        #noisy_stego_image = utils.adaptive_block_division(flow_gray,stego_image)
        noise= host - decoded
        noisy_stego = stego #+ noise #+host
        #noisy_stego_image = stego_image_quant + noise_tensor
        noisy_stego_dwt=dwt_module(noisy_stego)

        rs_dwt = decoder(noisy_stego_dwt)
        rs=iwt_module(rs_dwt)
        

        fnbit = loss_fn(rs, secret)
        mse=mse_loss(host, stego)
        mse_noisy = mse_loss(stego, noisy_stego)
        mse_low=mse_loss(host_dwt[:,:,:,0,:,:], stego_dwt[:,:,:,0,:,:])
        acc=utils.ACC(secret,rs)

        loss = 5*fnbit + mse + mse_low

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #print_decoder_gradients(decoder)

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
            
            #print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, MSE_NOISY: {avg_mse_noisy:.4f},FNBIT: {avg_fnbit:.4f}, Acc: {avg_acc:.4f}')
            #log_message = f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f},FNBIT: {avg_fnbit:.4f}, Acc: {avg_acc:.4f}'
            log_message = f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {avg_loss:.8f}, MSE: {avg_mse:.8f}, MSE_NOISY: {avg_mse_noisy:.4f},FNBIT: {avg_fnbit:.4f}, Acc: {avg_acc:.4f}'
            utils.log_to_file(log_message)
            print(log_message)
            #print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, FNBIT: {avg_fnbit:.4f}, Acc: {avg_acc:.4f}')
            # 重置累积值
            running_loss = 0.0
            running_mse = 0.0
            running_mse_noisy = 0.0
            running_fnbit = 0.0
            running_acc = 0.0
            batch_count = 0

            #torch.cuda.empty_cache()

       
        

save_dir = './tripdata_model_imporedEn_secRes_DenseDe_nonoisy/'
os.makedirs(save_dir, exist_ok=True)
encoder_save_path = os.path.join(save_dir, 'encoder_model.pth')
decoder_save_path = os.path.join(save_dir, 'decoder_model.pth')


torch.save(encoder.state_dict(), encoder_save_path)
torch.save(decoder.state_dict(), decoder_save_path)

print(f'Model saved ')
