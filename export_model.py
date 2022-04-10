import os
import argparse

import paddle

from network import get_networks
from utils.utils import get_config, load_pretrained_model

def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help="training configuration")
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the exported model',
        type=str,
        default='./output')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for export',
        type=str,
        default=None)

    return parser.parse_args()


def main(args, config):

    _, net = get_networks(config)

    if args.model_path is not None:
        load_pretrained_model(net, args.model_path)
        print('Loaded trained params of model successfully.')

    shape = [-1, 3, 128, 128]

    new_net = net

    new_net.eval()
    new_net = paddle.jit.to_static(
        new_net,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32'), 11, True])
    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(new_net, save_path)

    print(f'Model is saved in {args.save_dir}.')


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args.config)
    main(args, config)