import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import ffmpeg
import torch
import torch.nn as nn

class Vimeo90kDatasettxtNoisy(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sequences = self.load_sequences()

    def load_sequences(self):
        sequences = []
        train_list_path = os.path.join(self.root_dir, 'tri_trainlist.txt')
        sequences_dir = os.path.join(self.root_dir, 'sequences')
        
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sequence_path = os.path.join(sequences_dir, line)
                    sequences.append(sequence_path)
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_path = self.sequences[idx]
        images = []

        # 加载IM1, IM2
        for i in range(1, 3):
            img_path = os.path.join(sequence_path, f'im{i}.png')
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    raise FileNotFoundError(f"Image {img_path} cannot be read.")
            else:
                raise FileNotFoundError(f"Image {img_path} does not exist.")
        
        # 保存图像路径
        im1_path = os.path.join(sequence_path, 'im1.png')
        im2_path = os.path.join(sequence_path, 'im2.png')
        video_path = os.path.join(sequence_path, 'x265_32.mp4')
        decoded_im1_path = os.path.join(sequence_path, 'decoded_x265_32_im1.png')
        decoded_im2_path = os.path.join(sequence_path, 'decoded_x265_32_im2.png')
        decoded_im3_path = os.path.join(sequence_path, 'decoded_x265_32_im3.png')
        #noise_path = os.path.join(sequence_path, 'noise2_x265_32.npy')

        # 保存原始图像
        #Image.fromarray(images[0]).save(im1_path)
        #Image.fromarray(images[1]).save(im2_path)

        # Encode images into an H.265 video
        ffmpeg.input(f'{sequence_path}/im%d.png', framerate=1).output(video_path, vcodec='libx265', qp=32).run(overwrite_output=True, quiet=True)

        # Decode the video back into frames as decoded_im1.png and decoded_im2.png
        ffmpeg.input(video_path).output(f'{sequence_path}/decoded_x265_32_im%d.png', start_number=1).run(quiet=True)
        os.remove(decoded_im1_path)
        os.remove(decoded_im3_path)
        os.remove(video_path)

        # # 读取解码后的图像
        # decoded_im1 = cv2.imread(decoded_im1_path)
        # decoded_im2 = cv2.imread(decoded_im2_path)

        # if decoded_im1 is None or decoded_im2 is None:
        #     raise FileNotFoundError("Decoded images cannot be read.")

        # 计算解码后的图像与原始图像的差值
        #original_im2 = images[1]
        #decoded_im2 = cv2.cvtColor(decoded_im2, cv2.COLOR_BGR2RGB)
        # original_im2 = images[1].astype(np.int16)
        # decoded_im2 = cv2.cvtColor(decoded_im2, cv2.COLOR_BGR2RGB).astype(np.int16)
        #noise = original_im2 - decoded_im2

        #noise = noise.astype(np.int16)
        #np.save(noise_path, noise)
        #loaded_noise = np.load(noise_path).astype(np.int16)
        # 保存差值图像
        #noise = np.clip(noise, 0, 255).astype(np.uint8)
        #Image.fromarray(noise).save(noise_path)

        # mse = np.mean((original_im2 - decoded_im2) ** 2)
        # if mse == 0:
        #     psnr = float('inf')
        # else:
        #     max_pixel = 255.0
        #     psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

        return 1

dataset = Vimeo90kDatasettxtNoisy(root_dir='/home/admin/workspace/vimeo_triplet')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for num in dataloader:
    # 这里可以进行进一步的处理
    #print(im1.shape, im2.shape, decoded_im1.shape, decoded_im2.shape, noise.shape)
    pass