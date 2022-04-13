import os
import warnings
from argparse import ArgumentParser

import cv2
from dataset import pil_loader
import numpy as np
import paddle
from paddle.vision.transforms import Resize


from utils.utils import get_config, load_pretrained_model
from network import get_networks
from test_functions import grad_calc, gaussian_filter, convert_to_grayscale, morphological_process

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")

parser.add_argument('--image_path', type=str, default=None)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--save_dir', type=str, default='./output/')
parser.add_argument('--threshold', type=float, default=0.5)
warnings.filterwarnings('ignore')

def main():
    args = parser.parse_args()
    config = get_config(args.config)
    vgg, model = get_networks(config)

    if args.model_path is not None:
        load_pretrained_model(model, args.model_path)
    model.eval()
    transform = Resize([128, 128])
    origin_img = pil_loader(args.image_path)

    origin_img = transform(origin_img)
    img = np.array(origin_img).astype('float32') / 255

    img = paddle.to_tensor(img)
    img = img.transpose([2, 0, 1])
    img = img.unsqueeze(0)

    grad = grad_calc(img, model, vgg, config)
    grad_t = np.zeros((grad.shape[0], grad.shape[2], grad.shape[3]))
    for i in range(grad.shape[0]):
        grad_temp = convert_to_grayscale(grad[i].cpu().numpy())
        grad_temp = grad_temp.squeeze(0)
        grad_temp = gaussian_filter(grad_temp, sigma=4)
        grad_t[i] = grad_temp

    grad_t[grad_t < args.threshold] = 0
    grad_t = np.squeeze(grad_t, axis=0)
    heatmap = grad_t * 255
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    frame = np.array(origin_img)
    overlay = frame.copy()
    alpha = 0.5  # 设置覆盖图片的透明度
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), -1)  # 设置蓝色为热度图基本色
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # 将背景热度图覆盖到原图
    cv2.addWeighted(heatmap, alpha, frame, 1 - alpha, 0, frame)  # 将热度图覆盖到原图

    cv2.imwrite(os.path.join(args.save_dir, 'pred.png'), frame)
    pass





if __name__ == '__main__':
    main()