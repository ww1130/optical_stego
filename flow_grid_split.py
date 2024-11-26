# import cv2
# import numpy as np

# # 加载灰度图像
# image = cv2.imread('/mnt/workspace/vimeo_triplet/sequences/00024/0002/flow_im2_im3_gray.png', cv2.IMREAD_GRAYSCALE)

# # 计算X和Y方向上的梯度
# grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
# grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# # 计算梯度幅值
# grad_magnitude = cv2.magnitude(grad_x, grad_y)

# # 设定梯度阈值
# threshold = 200  # 可根据需要调整阈值

# # 创建一个空的掩码，用于绘制方块
# output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# def adaptive_block_division(image, x, y, width, height, threshold):
#     # 提取当前块的梯度幅值
#     block_grad = grad_magnitude[y:y + height, x:x + width]

#     # 判断是否需要进一步划分
#     if np.max(block_grad) > threshold and width > 8 and height > 8:  # 最小块大小改为16x16
#         half_width = width // 2
#         half_height = height // 2

#         # 递归划分四个子块
#         adaptive_block_division(image, x, y, half_width, half_height, threshold)
#         adaptive_block_division(image, x + half_width, y, half_width, half_height, threshold)
#         adaptive_block_division(image, x, y + half_height, half_width, half_height, threshold)
#         adaptive_block_division(image, x + half_width, y + half_height, half_width, half_height, threshold)
#     else:
#         # 如果不需要划分，则画出当前块
#         cv2.rectangle(output_image, (x, y), (x + width, y + height), (0, 255, 0), 1)


# # 将图像分割成32x32块，并对每个块进行自适应划分
# block_size = 64

# for y in range(0, image.shape[0], block_size):
#     for x in range(0, image.shape[1], block_size):
#         # 确保不要超出图像边界
#         width = min(block_size, image.shape[1] - x)
#         height = min(block_size, image.shape[0] - y)

#         # 对每个32x32的块进行自适应划分
#         adaptive_block_division(image, x, y, width, height, threshold)

# # 保存结果
# cv2.imwrite('/mnt/workspace/optical_stego/output_grid_image.png', output_image)


import cv2
import numpy as np

# 加载灰度图像
image_path = '/mnt/workspace/vimeo_triplet/sequences/00025/0002/flow_im2_im3_gray.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"无法加载图像: {image_path}")

# 计算X和Y方向上的梯度
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# 计算梯度幅值
grad_magnitude = cv2.magnitude(grad_x, grad_y)

# 设定两个梯度阈值
threshold_small = 180  # 用于64x64和32x32块的阈值
threshold_large = 500  # 用于更小块的阈值

# 创建一个空的掩码，用于绘制方块
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def adaptive_block_division(image, x, y, width, height, grad_magnitude, output_image, 
                           threshold_small, threshold_large, min_block_size):
    """
    自适应块划分函数，根据梯度幅值和块大小决定是否进一步划分。
    
    参数:
    - image: 原始灰度图像
    - x, y: 当前块的左上角坐标
    - width, height: 当前块的宽度和高度
    - grad_magnitude: 梯度幅值图
    - output_image: 输出图像，用于绘制矩形
    - threshold_small: 大块（64x64或32x32）的阈值
    - threshold_large: 小块（小于32x32）的阈值
    - min_block_size: 最小块大小，默认16
    """
    
    # 选择当前块应使用的阈值
    if width >= 32 :#and height >= 32:
        current_threshold = threshold_small
    else:
        current_threshold = threshold_large
    
    # 提取当前块的梯度幅值
    block_grad = grad_magnitude[y:y + height, x:x + width]

    # 判断是否需要进一步划分
    if np.max(block_grad) > current_threshold and width > min_block_size and height > min_block_size:
        half_width = width // 2
        half_height = height // 2

        # 递归划分四个子块
        adaptive_block_division(image, x, y, half_width, half_height, grad_magnitude, 
                               output_image, threshold_small, threshold_large, min_block_size)
        adaptive_block_division(image, x + half_width, y, half_width, half_height, grad_magnitude, 
                               output_image, threshold_small, threshold_large, min_block_size)
        adaptive_block_division(image, x, y + half_height, half_width, half_height, grad_magnitude, 
                               output_image, threshold_small, threshold_large, min_block_size)
        adaptive_block_division(image, x + half_width, y + half_height, half_width, half_height, grad_magnitude, 
                               output_image, threshold_small, threshold_large, min_block_size)
    else:
        # 如果不需要划分，则画出当前块
        cv2.rectangle(output_image, (x, y), (x + width, y + height), (0, 255, 0), 1)

# 定义初始块大小
initial_block_size = 64  # 可以调整为32或其他合适的值

# 将图像分割成初始块，并对每个块进行自适应划分
for y in range(0, image.shape[0], initial_block_size):
    for x in range(0, image.shape[1], initial_block_size):
        # 确保不要超出图像边界
        width = min(initial_block_size, image.shape[1] - x)
        height = min(initial_block_size, image.shape[0] - y)

        # 对每个初始块进行自适应划分
        adaptive_block_division(image, x, y, width, height, grad_magnitude, 
                               output_image, threshold_small, threshold_large, min_block_size=16)

# 保存结果
output_path = '/mnt/workspace/optical_stego/output_grid_image.png'
cv2.imwrite(output_path, output_image)
print(f"结果已保存到: {output_path}")
