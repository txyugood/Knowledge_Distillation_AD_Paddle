import os

import numpy as np

from dataset import ImageFolder
from paddle.vision.transforms import Compose, Resize
from paddle.io import DataLoader


def load_data(args, config):
    normal_class = config['normal_class']
    batch_size = config['batch_size']
    data_path = os.path.join(args.dataset_root, normal_class, 'train')

    mvtec_img_size = config['mvtec_img_size']

    orig_transform = Compose([
        Resize([mvtec_img_size, mvtec_img_size]),
    ])

    train_dataset = ImageFolder(root=data_path, transform=orig_transform)

    test_data_path = os.path.join(args.dataset_root, normal_class,'test')
    test_dataset = ImageFolder(root=test_data_path, transform=orig_transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    return train_dataloader, test_dataloader


def load_localization_data(args, config):
    normal_class = config['normal_class']
    mvtec_img_size = config['mvtec_img_size']

    orig_transform = Compose([
        Resize([mvtec_img_size, mvtec_img_size]),
    ])

    test_data_path = os.path.join(args.dataset_root, normal_class, 'test')
    test_set = ImageFolder(root=test_data_path, transform=orig_transform, localization_test=True)
    test_dataloader = DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
        num_workers=8,
    )

    ground_data_path = os.path.join(args.dataset_root, normal_class, 'ground_truth')
    ground_dataset = ImageFolder(root=ground_data_path, transform=orig_transform)
    ground_dataloader = DataLoader(
        ground_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=8,
    )

    x_ground = next(iter(ground_dataloader))[0].numpy()
    ground_temp = x_ground

    std_groud_temp = np.transpose(ground_temp, (0, 2, 3, 1))
    x_ground = std_groud_temp

    return test_dataloader, x_ground
