import torch
import os
import models
#from utils import dwt_transform, dwt_inverse,convert_to_tensor,StegoTensorProcessor
from dataset import Vimeo90kDatasettxtNoisytest
from torch.utils.data import DataLoader
import utils
from PIL import Image
import math
import numpy as np
import torch.nn as nn
import models
import ffmpeg
import cv2
import quant

ffmpeg_flag=False
ffmpeg_flag=True
# 定义损失函数
#criterion = StegoLosstest()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载数据集
dataset = Vimeo90kDatasettxtNoisytest(root_dir='/home/admin/workspace/vimeo_triplet')
#dataset = Vimeo90kDatasettxtNoisy(root_dir='/home/admin/workspace/vimeo_triplet')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# save_dir = './tripdata_model_imporedEn_secRes_DenseDe_10/'
save_dir = './tripdata_model_imporedEn_secRes_DenseDe_nonoisy_quant/'
#save_dir = './tripdata_model_imporedEn_secRes_DenseDe_nonoisy/'
save_dir = './tripdata_model_imporedEn_secRes_DenseDe_noisy_quant/'
encoder_save_path = os.path.join(save_dir, 'encoder_model.pth')
decoder_save_path = os.path.join(save_dir, 'decoder_model.pth')
# 训练过程
encoder= models.DenseEncoderNoisy().cuda()
decoder= models.DenseDecoderNoisy().cuda()

encoder.load_state_dict(torch.load(encoder_save_path))
decoder.load_state_dict(torch.load(decoder_save_path))
quantization = quant.Quantization().cuda()
# 将模型设置为评估模式
encoder.eval()
decoder.eval()

dwt_module = utils.DWT().cuda()
iwt_module = utils.IWT().cuda()
loss_fn = nn.BCEWithLogitsLoss()
# 测试过程
with torch.no_grad():
    for host123_no_norm, flow_gray in dataloader:
        #secret = torch.randint(0, 2, flow_gray.shape).float()
        #host=host2[:,1:,:,:,:]
        #host12进入视频编码，host23进入encoder
        #host12=host123[:,:2,:,:,:]
        host123_no_norm = host123_no_norm.permute(0, 1, 4, 2, 3)   
        host123 = host123_no_norm  / 127.5 - 1.0

        #host123=host123.permute(0, 1, 4, 2, 3)

        stego12=host123[:,:2,:,:,:].clone()
        host2=host123[:,1,:,:,:].unsqueeze(0)
        b,t,_,_,_=host2.shape

        secret = torch.randint(0, 2, (b, 1, 256, 448, 1), device=device).float()
        # secret = torch.randint(0, 2, host.shape).float()

        # DWT 变换
        secret = secret.permute(0, 1, 4, 2, 3)
        flow_gray = flow_gray.permute(0, 1, 4, 2, 3)
        flow_gray_dwt=dwt_module(flow_gray)

        secret_dwt = dwt_module(secret)
        host2_dwt = dwt_module(host2)
        
        res_dwt = encoder(host2_dwt, secret_dwt , flow_gray_dwt)

        res = iwt_module(res_dwt)
        #res=models.framenorm(res)

        stego=(host2+res).clamp(-1,1)
        stego_dwt = dwt_module(stego)
        stego_image=(stego+1.0)*127.5

        stego_quant=quantization(stego)

        if ffmpeg_flag:
            # stego12[:, 0, :, :, :]  =(stego12[:, 0, :, :, :]+1.0)*127.5
            #stego12里面的两张用于保存成stego_i.png+host3,后续进行视频编码
            stego12[:, 0, :, :, :]  =host123_no_norm[:, 0, :, :, :]
            stego12[:, 1, :, :, :] = stego_image.round()

            save_dir = "./video"
            os.makedirs(save_dir, exist_ok=True)
            stego_paths = [os.path.join(save_dir, f"stego{i+1}.png") for i in range(3)]
            flow_path=os.path.join(save_dir, "flow.png")
            video_path = os.path.join(save_dir, "stego_video.mp4")
            decoded_image_paths = [os.path.join(save_dir, f"video{i+1}.png") for i in range(2)]

            #for i in range(2):
            img_np = stego12[0, 0].permute(1, 2, 0).cpu().numpy()
            img_np = np.round(img_np).astype(np.uint8)
            Image.fromarray(img_np).save(stego_paths[0])

            img_np = stego12[0, 1].permute(1, 2, 0).cpu().numpy()#只有这一张是嵌入了信息的
            img_np = np.round(img_np).astype(np.uint8)
            Image.fromarray(img_np).save(stego_paths[1])

            img_np = host123_no_norm[0,2].permute(1, 2, 0).cpu().numpy()
            img_np = np.round(img_np).astype(np.uint8)
            Image.fromarray(img_np).save(stego_paths[2])
            
            img_np=((flow_gray[0,0,0]+1.0)*127.5).cpu().numpy()#.astype(np.uint8)#保存的光流图
            img_np = np.round(img_np).astype(np.uint8)
            Image.fromarray(img_np).save(flow_path)

            # Encode images into an H.265 video
            ffmpeg.input(f'{save_dir}/stego%d.png', framerate=1).output(video_path, vcodec='libx265', qp=22).run(overwrite_output=True,quiet=True)
            
            # Decode the video back into frames as png
            ffmpeg.input(video_path).output(f'{save_dir}/video%d.png', start_number=1).run(quiet=True)

            # Load video2.png as a tensor with shape (1, 1, 3, h, w)
            video2_img = cv2.imread(decoded_image_paths[1])
            video2_img = cv2.cvtColor(video2_img, cv2.COLOR_BGR2RGB)

            video2_tensor = torch.FloatTensor(video2_img).to(device)
            video2_tensor=video2_tensor.permute(2,0,1).unsqueeze(0).unsqueeze(0)
            #video2_tensor=video2_tensor.permute(0,2,1,3,4)
            video2_tensor=video2_tensor/127.5 -1.0
            
            video2_dwt_tensor=dwt_module(video2_tensor)

            rs_dwt = decoder(video2_dwt_tensor)
            rs=iwt_module(rs_dwt)

            #msebit = loss_fn(rs, secret)
            msestego=utils.MSE(host2, stego)
            msecompress=utils.MSE(stego,video2_tensor)
            acc=utils.ACC(secret,rs)

           
            psnrstego = 20 * math.log10(255.0 / math.sqrt(msestego))
            psnrcompress = 20 * math.log10(255.0 / math.sqrt(msecompress))
            print(f'psnrstego: {psnrstego}  psnrcompress: {psnrcompress} acc: { acc}')

        else:
            # noisy_stego_image = utils.adaptive_block_division(flow_gray,stego_image)
            #noise_guass = quant.noise(stego_image)
            #stego_image = stego_image + noise_guass
            #stego_image=stego_image.round()#######################,round后,再进入decoder，准确率降低#############################
            #stego_image_norm=stego_image/127.5 -1.0
            stego_image_norm_dwt=dwt_module(stego_quant)
            #noisy_stego_image = stego_image
            #noisy_stego_image= noisy_stego_image.round()
            #noisy_stego_image_tensor=dwt_module(noisy_stego_image)
            #rs_dwt = decoder(noisy_stego_image_tensor)
            rs_dwt = decoder(stego_image_norm_dwt)
            rs=iwt_module(rs_dwt)

            #fnbit = loss_fn(rs, secret)
            #msebit =utils.MSE(rs.sigmoid(),secret)
           
            #msestego=utils.MSE((host2+1.0)*127.5, stego_image)#计算round后的mse
            msestego=utils.MSE(host123_no_round[:,1,:,:,:], stego_image)#这两个值没有区别？
            acc=utils.ACC(secret,rs)
           
            psnrstego = 20 * math.log10(255.0 / math.sqrt(msestego))
            print(f'psnrstego: {psnrstego}  acc: { acc}')

        pass

print('Testing complete.')