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
import random
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

        # my fixs
        flow_grays = []
        for i in range(1, 7):
            flow_gray_path = os.path.join(sequence_path, f'flow_im{i}_im{i + 1}_gray.png')
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

# class Vimeo90kDatasettxt(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.sequences = self.load_sequences()

#     def load_sequences(self):
#         sequences = []
#         train_list_path = os.path.join(self.root_dir, 'sep_trainlist.txt')
#         sequences_dir = os.path.join(self.root_dir, 'sequences')
        
#         with open(train_list_path, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 line = line.strip()
#                 if line:
#                     sequence_path = os.path.join(sequences_dir, line)
#                     sequences.append(sequence_path)
#         return sequences

#     # def load_sequences(self, fraction=0.1):
#     #     sequences = []
#     #     train_list_path = os.path.join(self.root_dir, 'sep_trainlist.txt')
#     #     sequences_dir = os.path.join(self.root_dir, 'sequences')
        
#     #     with open(train_list_path, 'r') as f:
#     #         lines = f.readlines()
#     #         # 去除空白行
#     #         lines = [line.strip() for line in lines if line.strip()]
#     #         # 计算要选择的行数
#     #         num_lines_to_select = int(len(lines) * fraction)
#     #         # 确保选择的行数不超过总行数
#     #         if num_lines_to_select > len(lines):
#     #             num_lines_to_select = len(lines)
#     #         # 随机选择起始点
#     #         start_index = random.randint(0, len(lines) - num_lines_to_select)
#     #         # 从起始点开始选择连续的行
#     #         selected_lines = lines[start_index:start_index + num_lines_to_select]
#     #         for line in selected_lines:
#     #             sequence_path = os.path.join(sequences_dir, line)
#     #             sequences.append(sequence_path)
        
#     #     return sequences

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         sequence_path = self.sequences[idx]
#         images = []

#         # 加载7张连续的图片
#         for i in range(1, 8):
#             img_path = os.path.join(sequence_path, f'im{i}.png')
#             if os.path.exists(img_path):
#                 img = cv2.imread(img_path)
#                 if img is not None:
#                     images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#                 else:
#                     raise FileNotFoundError(f"Image {img_path} cannot be read.")
#             else:
#                 raise FileNotFoundError(f"Image {img_path} does not exist.")
        
#         # 转换为 tensor 格式
#         images = np.stack(images, axis=0)  # (7, H, W, C)

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         images_tensor = torch.FloatTensor(images).to(device)
        
#         return images_tensor


# class Vimeo90kDatasettxt(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.sequences = self.load_sequences()

#     def load_sequences(self):
#         sequences = []
#         train_list_path = os.path.join(self.root_dir, 'sep_trainlist.txt')
#         sequences_dir = os.path.join(self.root_dir, 'sequences')
        
#         with open(train_list_path, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 line = line.strip()
#                 if line:
#                     sequence_path = os.path.join(sequences_dir, line)
#                     sequences.append(sequence_path)
#         return sequences

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         sequence_path = self.sequences[idx]
#         images = []

#         # 加载7张连续的图片
#         for i in range(1, 8):
#             img_path = os.path.join(sequence_path, f'im{i}.png')
#             if os.path.exists(img_path):
#                 img = cv2.imread(img_path)
#                 if img is not None:
#                     images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#                 else:
#                     raise FileNotFoundError(f"Image {img_path} cannot be read.")
#             else:
#                 raise FileNotFoundError(f"Image {img_path} does not exist.")
        
#         # 加载光流灰度图
#         flow_grays = []
#         for i in range(1, 7):
#             flow_gray_path = os.path.join(sequence_path, f'flow_im{i}_im{i+1}_gray.png') 
#             if os.path.exists(flow_gray_path):
#                 flow_gray = cv2.imread(flow_gray_path, cv2.IMREAD_GRAYSCALE)
#                 if flow_gray is not None:
#                     flow_grays.append(flow_gray)
#             else:
#                 raise FileNotFoundError(f"Flow gray image {flow_gray_path} does not exist.")
        
#         # 转换为 tensor 格式
#         images = np.stack(images, axis=0)  # (7, H, W, C)
#         flow_grays = np.stack(flow_grays, axis=0)  # (6, H, W)
#         flow_grays = np.expand_dims(flow_grays, axis=-1)  # (6, H, W, 1)

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         images_tensor = torch.FloatTensor(images).to(device)
#         flow_grays_tensor = torch.FloatTensor(flow_grays).to(device)
        
#         return images_tensor, flow_grays_tensor
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
        decoded_im2_path = os.path.join(sequence_path, 'decoded_x265_37_im2.png')
        #print(sequence_path)
        images = []

        # 加载IM2
        for i in range(2, 3):
            img_path = os.path.join(sequence_path, f'im{i}.png')
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    raise FileNotFoundError(f"Image {img_path} cannot be read.")
            else:
                raise FileNotFoundError(f"Image {img_path} does not exist.")
        
        # 加载光流灰度图FLOW23
        flow_grays = []
        for i in range(2, 3):
            flow_gray_path = os.path.join(sequence_path, f'flow_im{i}_im{i+1}_gray.png') 
            if os.path.exists(flow_gray_path):
                flow_gray = cv2.imread(flow_gray_path, cv2.IMREAD_GRAYSCALE)
                if flow_gray is not None:
                    flow_grays.append(flow_gray)
            else:
                raise FileNotFoundError(f"Flow gray image {flow_gray_path} does not exist.")
        
        # 转换为 tensor 格式
        images = np.stack(images, axis=0)  # (3, H, W, C)
        
        flow_grays = np.stack(flow_grays, axis=0)  # (3, H, W)
        flow_grays = np.expand_dims(flow_grays, axis=-1)  # (6, H, W, 1)

        decoded_im2 = cv2.imread(decoded_im2_path)
        decoded_im2 = cv2.cvtColor(decoded_im2, cv2.COLOR_BGR2RGB)

        images = images / 127.5 - 1.0
        flow_grays = flow_grays / 127.5 - 1.0
        decoded_im2 = decoded_im2 / 127.5 - 1.0


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        images_tensor = torch.FloatTensor(images).to(device)
        flow_grays_tensor = torch.FloatTensor(flow_grays).to(device)
        #flow_grays_tensor=flow_grays_tensor*0.1
        decoded_im2_tensor = torch.FloatTensor(decoded_im2).to(device)
        decoded_im2_tensor=decoded_im2_tensor.unsqueeze(0)
        
        return images_tensor, flow_grays_tensor,decoded_im2_tensor

#编码获取真实噪声，需要加载三张im
class Vimeo90kDatasettxtNoisyReal(Dataset):
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
        #decoded_im2_path = os.path.join(sequence_path, 'decoded_x265_37_im2.png')
        #print(sequence_path)
        images = []

        # 加载IM1,2,3
        for i in range(1, 4):
            img_path = os.path.join(sequence_path, f'im{i}.png')
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    raise FileNotFoundError(f"Image {img_path} cannot be read.")
            else:
                raise FileNotFoundError(f"Image {img_path} does not exist.")
        
        # 加载光流灰度图FLOW23
        flow_grays = []
        for i in range(2, 3):
            flow_gray_path = os.path.join(sequence_path, f'flow_im{i}_im{i+1}_gray.png') 
            if os.path.exists(flow_gray_path):
                flow_gray = cv2.imread(flow_gray_path, cv2.IMREAD_GRAYSCALE)
                if flow_gray is not None:
                    flow_grays.append(flow_gray)
            else:
                raise FileNotFoundError(f"Flow gray image {flow_gray_path} does not exist.")
        
        # 转换为 tensor 格式
        images = np.stack(images, axis=0)  # (3, H, W, C)
        
        flow_grays = np.stack(flow_grays, axis=0)  # (3, H, W)
        flow_grays = np.expand_dims(flow_grays, axis=-1)  # (6, H, W, 1)

        #decoded_im2 = cv2.imread(decoded_im2_path)
        #decoded_im2 = cv2.cvtColor(decoded_im2, cv2.COLOR_BGR2RGB)

        #images = images / 127.5 - 1.0###先不做归一化，因为后面要把原始的图片送入ffmpeg进行编码
        flow_grays = flow_grays / 127.5 - 1.0
        #decoded_im2 = decoded_im2 / 127.5 - 1.0


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        images_tensor = torch.FloatTensor(images).to(device)
        flow_grays_tensor = torch.FloatTensor(flow_grays).to(device)
        #flow_grays_tensor=flow_grays_tensor*0.1
        #decoded_im2_tensor = torch.FloatTensor(decoded_im2).to(device)
        #decoded_im2_tensor=decoded_im2_tensor.unsqueeze(0)
        
        return images_tensor, flow_grays_tensor#,decoded_im2_tensor


class Vimeo90kDatasettxtNoisytest(Dataset):
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

        #print(sequence_path)

        # 加载IM1,IM2,im3,test时，encoder im2,视频编码im1.im2,im3得到视频
        for i in range(1, 4):
            img_path = os.path.join(sequence_path, f'im{i}.png')
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    raise FileNotFoundError(f"Image {img_path} cannot be read.")
            else:
                raise FileNotFoundError(f"Image {img_path} does not exist.")
        
        # 加载光流灰度图FLOW23
        flow_grays = []
        for i in range(2, 3):
            flow_gray_path = os.path.join(sequence_path, f'flow_im{i}_im{i+1}_gray.png') 
            if os.path.exists(flow_gray_path):
                flow_gray = cv2.imread(flow_gray_path, cv2.IMREAD_GRAYSCALE)
                if flow_gray is not None:
                    flow_grays.append(flow_gray)
            else:
                raise FileNotFoundError(f"Flow gray image {flow_gray_path} does not exist.")
        
        # 转换为 tensor 格式
        images = np.stack(images, axis=0)  # (3, H, W, C)
        flow_grays = np.stack(flow_grays, axis=0)  # (3, H, W)
        flow_grays = np.expand_dims(flow_grays, axis=-1)  # (6, H, W, 1)

        #images = images / 127.5 - 1.0  #在这里先不做round,因为test时，计算psnr需要和原始的图片计算
        flow_grays = flow_grays / 127.5 - 1.0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images_tensor = torch.FloatTensor(images).to(device)
        flow_grays_tensor = torch.FloatTensor(flow_grays).to(device)
        #flow_grays_tensor=flow_grays_tensor*0.1
        
        return images_tensor, flow_grays_tensor