import torch
import torch.nn as nn
from models import DenseDecoder,DenseEncoder

model = DenseEncoder()

# 从文件中加载模型参数
model.load_state_dict(torch.load('./saved_models_whole_data/encoder_model.pth'))

# 统计模型参数
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

total_params, trainable_params = count_parameters(model)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")