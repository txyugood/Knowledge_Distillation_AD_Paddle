import os
from argparse import ArgumentParser
import pickle

import paddle
from dataset import ImageFolder
from paddle.vision.transforms import Compose, Resize
from paddle.io import DataLoader

from utils.utils import get_config
from network import get_networks
from loss_functions import MseDirectionLoss, DirectionOnlyLoss
from test_functions import detection_test

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")
parser.add_argument('--dataset_root', type=str, default=None,)

def train(args, config):
    direction_loss_only = config["direction_loss_only"]
    normal_class = config["normal_class"]
    learning_rate = float(config['learning_rate'])
    num_epochs = config["num_epochs"]
    lamda = config['lamda']
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
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    vgg, model = get_networks(config)

    if direction_loss_only:
        criterion = DirectionOnlyLoss()
    else:
        criterion = MseDirectionLoss(lamda)

    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=learning_rate)
    losses = []
    roc_aucs = []
    best_roc_auc = 0
    for epoch in range(num_epochs + 1):
        model.train()
        epoch_loss = 0
        for data in train_dataloader:
            X = data[0]
            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)

            output_pred = model.forward(X)
            with paddle.no_grad():
                output_real = vgg(X)

            total_loss = criterion(output_pred, output_real)

            # Add loss to the list
            epoch_loss += total_loss.item()
            losses.append(total_loss.item())

            # Compute gradients
            total_loss.backward()
            # Adjust weights
            optimizer.step()
            model.clear_gradients()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        if epoch % 10 == 0:
            roc_auc = detection_test(model, vgg, test_dataloader, config)
            roc_aucs.append(roc_auc)
            print("RocAUC at epoch {}:".format(epoch), roc_auc)
        if roc_auc > best_roc_auc:
            best_roc_auc = best_roc_auc
            paddle.save(model.state_dict(),
                       '{}Cloner_{}_epoch_{}.pth'.format('./output', normal_class, epoch))
            paddle.save(optimizer.state_dict(),
                       '{}Opt_{}_epoch_{}.pth'.format('./output', normal_class, epoch))


        # if epoch % 50 == 0:
        #     torch.save(model.state_dict(),
        #                '{}Cloner_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, epoch))
        #     torch.save(optimizer.state_dict(),
        #                '{}Opt_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, epoch))
        #     with open('{}Auc_{}_epoch_{}.pickle'.format(checkpoint_path, normal_class, epoch),
        #               'wb') as f:
        #         pickle.dump(roc_aucs, f)

def main():
    args = parser.parse_args()
    config = get_config(args.config)
    train(args, config)


if __name__ == '__main__':
    main()
