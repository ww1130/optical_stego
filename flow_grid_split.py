import cv2
import numpy as np

# 加载灰度图像
image = cv2.imread('/home/user/ww/vimeo90k/sequences/00026/0001/flow_im1_im2_gray.png', cv2.IMREAD_GRAYSCALE)

# 计算X和Y方向上的梯度
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# 计算梯度幅值
grad_magnitude = cv2.magnitude(grad_x, grad_y)

# 设定梯度阈值
threshold = 150  # 可根据需要调整阈值

# 创建一个空的掩码，用于绘制方块
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


# 自适应块划分函数
# def adaptive_block_division(image, x, y, width, height, threshold):
#     # 提取当前块的梯度幅值
#     block_grad = grad_magnitude[y:y + height, x:x + width]
#
#     # 判断是否需要进一步划分
#     if np.max(block_grad) > threshold and width > 8 and height > 8:  # 最小块大小为8x8
#         half_width = width // 2
#         half_height = height // 2
#
#         # 递归划分四个子块
#         adaptive_block_division(image, x, y, half_width, half_height, threshold)
#         adaptive_block_division(image, x + half_width, y, half_width, half_height, threshold)
#         adaptive_block_division(image, x, y + half_height, half_width, half_height, threshold)
#         adaptive_block_division(image, x + half_width, y + half_height, half_width, half_height, threshold)
#     else:
#         # 如果不需要划分，则画出当前块
#         cv2.rectangle(output_image, (x, y), (x + width, y + height), (0, 255, 0), 1)

def adaptive_block_division(image, x, y, width, height, threshold):
    # 提取当前块的梯度幅值
    block_grad = grad_magnitude[y:y + height, x:x + width]

    # 判断是否需要进一步划分
    if np.max(block_grad) > threshold and width > 16 and height > 16:  # 最小块大小改为16x16
        half_width = width // 2
        half_height = height // 2

        # 递归划分四个子块
        adaptive_block_division(image, x, y, half_width, half_height, threshold)
        adaptive_block_division(image, x + half_width, y, half_width, half_height, threshold)
        adaptive_block_division(image, x, y + half_height, half_width, half_height, threshold)
        adaptive_block_division(image, x + half_width, y + half_height, half_width, half_height, threshold)
    else:
        # 如果不需要划分，则画出当前块
        cv2.rectangle(output_image, (x, y), (x + width, y + height), (0, 255, 0), 1)


# 将图像分割成32x32块，并对每个块进行自适应划分
block_size = 64

for y in range(0, image.shape[0], block_size):
    for x in range(0, image.shape[1], block_size):
        # 确保不要超出图像边界
        width = min(block_size, image.shape[1] - x)
        height = min(block_size, image.shape[0] - y)

        # 对每个32x32的块进行自适应划分
        adaptive_block_division(image, x, y, width, height, threshold)

# 保存结果
cv2.imwrite('/home/user/ww/mmflow/demo/output_grid_image.png', output_image)
