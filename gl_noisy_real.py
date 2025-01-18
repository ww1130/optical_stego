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
from dataset import Vimeo90kDatasettxtNoisy,Vimeo90kDatasettxtNoisyReal
import ffmpeg
import time


save_dir = './tripdata_model_imporedEn_secRes_DenseDe_nonoisy/'
encoder_save_path = os.path.join(save_dir, 'encoder_model.pth')
decoder_save_path = os.path.join(save_dir, 'decoder_model.pth')


save_dir = './tripdata_model_yuv/'
encoder_save_path = os.path.join(save_dir, 'encoder_model_264_32_07.pth')
decoder_save_path = os.path.join(save_dir, 'decoder_model_264_32_07.pth')

encoder_save_path = os.path.join(save_dir, 'encoder_model_264_32_10_crf32.pth')
decoder_save_path = os.path.join(save_dir, 'decoder_model_264_32_10_crf32.pth')


log_file_path = 'train.log'
with open(log_file_path, 'w') as f:
    pass  # 这里什么都不做，只是清空文件

# 加载数据集

width, height = 448, 256
target_mse_low=1e-5
bs=64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs=2
print_every_batch=1
generate_secret_every_batch=1
dataset = Vimeo90kDatasettxtNoisyReal(root_dir='/home/admin/workspace/vimeo_triplet')
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True,drop_last=True)
quantization = quant.Quantization().cuda()
# 训练过程
encoder= models.DenseEncoderNoisy().cuda()
decoder= models.DenseDecoderNoisy().cuda()
# encoder.load_state_dict(torch.load(encoder_save_path))
# decoder.load_state_dict(torch.load(decoder_save_path))

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=1e-4)
#optimizer = torch.optim.Adam(decoder.parameters(),lr=1e-4)#
encoder_mse_threshold_low = 6  # 当 MSE 小于此值时，loss=fnbit
encoder_mse_threshold_high = 8 # 当 MSE 大于此值时，loss=mse+fnbit

loss_fn = nn.BCEWithLogitsLoss().cuda()
#loss_fn = nn.BCELoss().cuda()
mse_loss = nn.MSELoss().to(device)
dwt_module = utils.DWT().cuda()
iwt_module = utils.IWT().cuda()
alpha=0.0
lambda2=0.0
count=0

for epoch in range(epochs):
    running_loss = 0.0
    running_mse_quant=0.0
    running_mse = 0.0
    running_mse_noisy=0.0
    running_fnbit = 0.0
    running_acc = 0.0
    batch_count = 0  # 用于计数当前epoch中的batch数

    # secret = torch.randint(0, 2, (64, 1, 256, 448, 1), device=device).float()
    # secret = secret.permute(0, 1, 4, 2, 3)
    # secret_dwt = dwt_module(secret)

    for batch_idx, (host123,flow_gray) in enumerate(dataloader):
        b,t,h,w,_=host123.shape
        #host123=host123.permute(0, 1, 4, 2, 3)#转换为b,t,c,h,w
        #host2=host123[:,1:2,:,:,:]
        # 初始化用于保存所有批次结果的列表
        y_list, u_list, v_list = [], [], []
        for bidx in range(b):
            images_rgb = [host123[bidx, 0].cpu().numpy(),host123[bidx, 1].cpu().numpy(), host123[bidx, 2].cpu().numpy()]
            images_rgb = [np.round(img).astype(np.uint8) for img in images_rgb]
            y_time, u_time, v_time = utils.rgb_to_yuv420(images_rgb)#转换成yuv420
            y_list.append(y_time.unsqueeze(0))
            u_list.append(u_time.unsqueeze(0))
            v_list.append(v_time.unsqueeze(0))
        y_all = torch.cat(y_list, dim=0)
        u_all = torch.cat(u_list, dim=0)
        v_all = torch.cat(v_list, dim=0)
        #utils.save_yuv_sequence(y_all, u_all, v_all)#保存yuv

        #取出第二帧
        y=y_all[:,1:2,:,:].clone().unsqueeze(2)
        u=u_all[:,1:2,:,:].clone().unsqueeze(2)
        v=v_all[:,1:2,:,:].clone().unsqueeze(2)

        #对第二帧进行归一化 -1,1
        y_norm=y/127.5-1.0
        u_norm=u/127.5-1.0
        v_norm=v/127.5-1.0

        #对y分量做dwt   #拼接y，u，v
        y_norm_dwt=dwt_module(y_norm).squeeze(1)
        u_norm_dwt=dwt_module(u_norm).squeeze(1)
        v_norm_dwt=dwt_module(v_norm).squeeze(1)
        yuv_norm_dwt = torch.cat([y_norm_dwt,u_norm_dwt,v_norm_dwt],dim=2).unsqueeze(1)
        yuv_norm_dwt = yuv_norm_dwt.reshape(b,1,3,4,128,224)
        #y_norm_dwt_uvnorm=torch.cat([y_norm_dwt,u_norm,v_norm],dim=2).unsqueeze(1)

        # y_norm_dwt_iwt=iwt_module(y_norm_dwt.unsqueeze(1))
        # y_norm_dwt_iwt_unorm=(y_norm_dwt_iwt+1)*127.5
        # mse= mse_loss(y_norm_dwt_iwt_unorm,y)

        secret = torch.randint(0, 2, (b, 1, 256, 448, 1), device=device).float()
        secret = secret.permute(0, 1, 4, 2, 3)
        secret_dwt = dwt_module(secret)

        #host_norm = host_norm.permute(0, 1, 4, 2, 3)
        #host_norm=host_norm[:,1,:,:,:].unsqueeze(1)
        flow_gray=flow_gray.permute(0, 1, 4, 2, 3)
        flow_gray_dwt = dwt_module(flow_gray)
        res_dwt = encoder(yuv_norm_dwt, secret_dwt ,flow_gray_dwt)#出来的形状是(b,t,1,1,6,h/2,w/2),

        res=iwt_module(res_dwt)#
        stego_dwt = (yuv_norm_dwt + res_dwt)#.clamp(-1,1)#这里是归一化后的
        stego=iwt_module(stego_dwt).clamp(-1,1)
        #stego_uv=torch.cat((u_norm+res_dwt[:,:,:,4:5,:,:].squeeze(1),v_norm+res_dwt[:,:,:,5:6,:,:].squeeze(1)),dim=2).clamp(-1,1)
        #stego_dwt=torch.cat([stego_y_dwt,stego_uv.unsqueeze(1)],dim=3)

        # stego_dwt=y_norm_dwt_uvnorm+res_dwt#(b,t,1,1,6,h/2,w/2),y分量是以频带形式存在的
        # stego_y_dwt = stego_dwt[:,:,:,0:4,:,:]#提取出y分量，y分量是以频带形式存在的
        # stego_y=iwt_module(stego_y_dwt).clamp(-1,1)#把y分量进行iwt变换后，再clamp
        #stego_quant=quantization(stego)#64,1,1,h,w
        
        #这个用于进行视频编码和计算像素级别的损失
        stego_255=((stego+1.0)*127.5).round()
        

        #把第二帧替换成stego，保存为yuv，再进行编码
        y_all[:,1,:,:]=stego_255[:,0,0,:,:]
        u_all[:,1,:,:]=stego_255[:,0,1,:,:]
        v_all[:,1,:,:]=stego_255[:,0,2,:,:]
        utils.save_yuv_sequence(y_all, u_all, v_all)#保存yuv
        #noisy_stego_image = utils.adaptive_block_division(flow_gray,stego_image)

        #time.sleep(1)
        #final_yuv=stego_255
        final_y, final_u, final_v = utils.process_batch(b, 'output', width=448, height=256,qp=32)
        final_y=final_y.unsqueeze(1).unsqueeze(1)
        final_u=final_u.unsqueeze(1).unsqueeze(1)
        final_v=final_v.unsqueeze(1).unsqueeze(1)
        final_yuv=torch.cat([final_y,final_u,final_v],dim=2)

        final_yuv_norm=final_yuv/127.5 -1.0
        final_yuv_norm_dwt=dwt_module(final_yuv_norm)

        #final_y_norm_dwt=dwt_module(final_y_norm).squeeze(1)
        #final_y_norm_dwt_uvnorm=torch.cat([final_y_norm_dwt, final_u_norm.unsqueeze(1), final_v_norm.unsqueeze(1)],dim=2).unsqueeze(1)

        #noise=final_y_norm_dwt_uvnorm-stego_dwt
        noise=alpha*(final_yuv_norm_dwt-stego_dwt).detach()
        #stego_noise=(stego_255 +  noise).clamp(0,255)
        #stego_noise_dwt= dwt_module(stego_noise/127.5-1.0 )
        stego_noise_dwt = stego_dwt + noise
        #utils.process_batch(b, 'output', width, height)

        rs_dwt = decoder(stego_noise_dwt,flow_gray_dwt)
        rs=iwt_module(rs_dwt)#.clamp(0,1)#要加clamp吗

        fnbit = loss_fn(rs, secret)#哪个在前面
        #fnbit=mse_loss(rs,secret)
        #fnbit = loss_fn(rs_dwt, secret_dwt)
        #fnbit = mse_loss(rs_dwt, secret_dwt)

        #应该改成计算mse_stego?
        mse_y_low=mse_loss(stego_dwt[:,:,0,0:1,:,:], y_norm_dwt[:,:,0:1,:,:]) 
        #mse_y_high=mse_loss(stego_dwt[:,:,0,1:4,:,:], y_norm_dwt[:,:,1:4,:,:])

        mse_u_low=mse_loss(stego_dwt[:,:,1,0:1,:,:], u_norm_dwt[:,:,0:1,:,:])
        #mse_u_high=mse_loss(stego_dwt[:,:,1,1:4,:,:], u_norm_dwt[:,:,1:4,:,:])

        mse_v_low=mse_loss(stego_dwt[:,:,2,0:1,:,:], v_norm_dwt[:,:,0:1,:,:])
        #mse_v_high=mse_loss(stego_dwt[:,:,2,1:4,:,:], v_norm_dwt[:,:,1:4,:,:])

        mse_y_norm=mse_loss(stego[:,:,0:1,:,:],y_norm)
        mse_u_norm=mse_loss(stego[:,:,1:2,:,:],u_norm)
        mse_v_norm=mse_loss(stego[:,:,2:3,:,:],v_norm)
        #mse_all=mse_loss(stego_noise,)


        #loss = fnbit + (mse_y_low+mse_y_norm)/( ((mse_y_low+mse_y_norm)/fnbit).detach() ) + ( mse_u_low + mse_u_norm  + mse_v_low + mse_v_norm)/(( (mse_u_low + mse_u_norm  + mse_v_low + mse_v_norm)/fnbit).detach())
        #loss = fnbit + 3*( mse_y_low+mse_y_norm ) + 2*( mse_u_low + mse_u_norm  + mse_v_low + mse_v_norm)
        #07以前的系数是132,mse均在几十并且acc为0.99，之后y分量mse正常，但是uv分量失真大，需要调整为1，3，2.5？
        # loss = fnbit + 3*( mse_y_low+mse_y_norm ) + 2.5*( mse_u_low + mse_u_norm  + mse_v_low + mse_v_norm)
        loss = fnbit + 1.1*lambda2*( mse_y_low+mse_y_norm ) + lambda2*( mse_u_low + mse_u_norm  + mse_v_low + mse_v_norm)
        #loss=fnbit
        # if(fnbit>0.3):
        #     loss=fnbit
        # else:
        #     loss=mse_y_low+mse_y_norm+mse_u_low + mse_u_norm  + mse_v_low + mse_v_norm
        
        acc=utils.ACC(secret,rs)
        #隐写的mse
        mse_pixel_y=mse_loss(stego_255[:,:,0:1,:,:],y).item()                                                                                         
        mse_pixel_u=mse_loss(stego_255[:,:,1:2,:,:],u).item()
        mse_pixel_v=mse_loss(stego_255[:,:,2:3,:,:],v).item()
        #视频编解码的mse
        mse_video_code_y=mse_loss(stego_255[:,:,0:1,:,:],final_yuv[:,:,0:1,:,:]).item()
        mse_video_code_u=mse_loss(stego_255[:,:,1:2,:,:],final_yuv[:,:,1:2,:,:]).item()
        mse_video_code_v=mse_loss(stego_255[:,:,2:3,:,:],final_yuv[:,:,2:3,:,:]).item()

        if(acc>0.95):
            alpha=min(1.0,alpha+0.005)
            lambda2+=0.015

        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
        #utils.print_decoder_gradients(encoder)

       
        # if(acc>0.98 and mse_pixel_y < 30):
        #     alpha= min(alpha+0.01,1.0)
        #alpha=torch.empty(1).uniform_(0.95, 1.01).item()
        #alpha=0.8
        
        # 每32个batch打印一次
        if (batch_idx + 1) % print_every_batch == 0:
            #print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, MSE_NOISY: {avg_mse_noisy:.4f},FNBIT: {avg_fnbit:.4f}, Acc: {avg_acc:.4f}')
            #log_message = f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f},FNBIT: {avg_fnbit:.4f}, Acc: {avg_acc:.4f}'
            log_message = (f'E{epoch + 1},B{batch_idx + 1},'
            f'L:{loss.item():.2f},fn:{fnbit.item():.2f}, ylow:{mse_y_low.item():.4f},ulow:{mse_u_low.item():.4f},vlow:{mse_v_low.item():.4f}, '
            f'y:{mse_y_norm.item():.4f},u:{mse_u_norm.item():.4f},v:{mse_v_norm.item():.4f}, '
            f'p_y:{mse_pixel_y:.0f},p_u:{mse_pixel_u:.0f},p_v:{mse_pixel_v:.0f}, '
            f'c_y:{mse_video_code_y:.0f},c_u:{mse_video_code_u:.0f},c_v:{mse_video_code_v:.0f},alpha:{alpha:.2f},lambda2:{lambda2:.2f}'
            f' acc: {acc:.4f}')
            utils.log_to_file(log_message)
            print(log_message)
            pass
            #print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, FNBIT: {avg_fnbit:.4f}, Acc: {avg_acc:.4f}')
            # 重置累积值
            # running_loss = 0.0
            # running_mse_quant=0.0
            # running_mse = 0.0
            # running_mse_noisy = 0.0
            # running_fnbit = 0.0
            # running_acc = 0.0
            # batch_count = 0

            #torch.cuda.empty_cache()

       
        

save_dir = './tripdata_model_yuv/'
os.makedirs(save_dir, exist_ok=True)
encoder_save_path = os.path.join(save_dir, 'encoder_model_264_32_10_crf32_y11.pth')
decoder_save_path = os.path.join(save_dir, 'decoder_model_264_32_10_crf32_y11.pth')


torch.save(encoder.state_dict(), encoder_save_path)
torch.save(decoder.state_dict(), decoder_save_path)

print(f'Model saved ')
