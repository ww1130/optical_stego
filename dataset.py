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
