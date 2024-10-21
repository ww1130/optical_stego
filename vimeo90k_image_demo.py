import os
import os.path as osp
from argparse import ArgumentParser
import mmcv
import re
import numpy as np
import cv2
from mmflow.apis import inference_model, init_model
from mmflow.datasets import write_flow


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--valid',
                        help='Valid file. If the predicted flow is sparse, valid mask will filter the output flow map.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def read_flo_file(filename):
    """Reads a .flo file in Middlebury format."""
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, dtype=np.float32, count=1)
        if magic != 202021.25:
            raise ValueError('Invalid .flo file')

        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]

        flow_data = np.fromfile(f, dtype=np.float32, count=2 * width * height)
        flow_data = np.resize(flow_data, (height, width, 2))

    return flow_data


def compute_flow_speed(flow):
    """Computes flow speed from the flow data."""
    flow_u = flow[:, :, 0]
    flow_v = flow[:, :, 1]
    flow_speed = np.sqrt(flow_u ** 2 + flow_v ** 2)
    return flow_speed


def save_flow_speed_as_grayscale(flow_speed, save_file):
    """Saves flow speed as a grayscale image."""
    normalized_flow_speed = cv2.normalize(flow_speed, None, 0, 255, cv2.NORM_MINMAX)
    gray_image = np.uint8(normalized_flow_speed)
    cv2.imwrite(save_file, gray_image)


def process_sequence(model, sequence_dir, device):
    pattern = r'^im\d\.png$'
    images = sorted([os.path.join(sequence_dir, img) for img in os.listdir(sequence_dir) if re.match(pattern, img)])

    for i in range(len(images) - 1):
        img1 = images[i]
        img2 = images[i + 1]

        # 计算光流
        result = inference_model(model, img1, img2, valids=args.valid)

        # 生成文件名前缀
        out_prefix = f'flow_{osp.basename(img1)[:-4]}_{osp.basename(img2)[:-4]}'
        flow_flo_path = osp.join(sequence_dir, f'{out_prefix}.flo')

        # 保存 .flo 文件
        write_flow(result, flow_flo_path)

        # 读取并转换为灰度图
        flow_data = read_flo_file(flow_flo_path)
        flow_speed = compute_flow_speed(flow_data)

        # 保存灰度图
        flow_gray_path = osp.join(sequence_dir, f'{out_prefix}_gray.png')
        save_flow_speed_as_grayscale(flow_speed, flow_gray_path)

        # 删除 .flo 文件
        os.remove(flow_flo_path)

        #print(f'Successfully saved {flow_gray_path} and deleted {flow_flo_path}')


def main(args):
    # 初始化模型
    model = init_model(args.config, args.checkpoint, device=args.device)
    data_dir = 'data'

    # 遍历 vimeo90k 的文件结构
    for root, dirs, files in os.walk(data_dir):
        # dirs = [
        #     "00003", "00077", "00039", "00007", "00002", "00054", "00064", "00051",
        #     "00024", "00019", "00091", "00001", "00012", "00038", "00073", "00084", "00045",
        #     "00028", "00058", "00032", "00048", "00086", "00063", "00029", "00014", "00057",
        #     "00041", "00065"
        # ]
        print(root, dirs, files)
        for dir_name in dirs:
            sequence_dir = osp.join(root, dir_name)
            for root2, dirs2, files2 in os.walk(sequence_dir):
                for dir_name2 in dirs2:
                    sequence_dir2 = osp.join(root2, dir_name2)
                    if len(os.listdir(sequence_dir2)) >= 7:
                        process_sequence(model, sequence_dir2, args.device)

            print(f'Successfully processed {sequence_dir}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
