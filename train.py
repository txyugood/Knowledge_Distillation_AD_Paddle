import os
import warnings
from argparse import ArgumentParser
import pickle

import paddle

from utils.utils import get_config
from network import get_networks
from dataloader import load_data
from loss_functions import MseDirectionLoss, DirectionOnlyLoss
from test_functions import detection_test

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")
parser.add_argument('--dataset_root', type=str, default=None)
parser.add_argument('--normal_class', type=str, default='capsule')
parser.add_argument('--save_dir', type=str, default='./output/')
warnings.filterwarnings('ignore')

def train(args, config):
    direction_loss_only = config["direction_loss_only"]
    normal_class = config["normal_class"]
    learning_rate = float(config['learning_rate'])
    num_epochs = config["num_epochs"]
    lamda = config['lamda']

    train_dataloader, test_dataloader = load_data(args, config)
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

        print('[Train] epoch [{}/{}], loss:{:.4f} class:{}'.format(epoch, num_epochs, epoch_loss, normal_class))
        if (epoch % 10 == 0 and epoch != 0) or epoch == num_epochs:
            roc_auc = detection_test(model, vgg, test_dataloader, config)
            roc_aucs.append(roc_auc)
            print("[Eval] {} class RocAUC at epoch {}:".format(normal_class, epoch), roc_auc)
            if roc_auc > best_roc_auc:
                print(f"[Eval] save best model at epoch {epoch}")
                os.makedirs(f"./output/{normal_class}", exist_ok=True)
                best_roc_auc = roc_auc
                paddle.save(model.state_dict(), os.path.join(args.save_dir, f'{normal_class}/best_model.pdparams'))
                paddle.save(optimizer.state_dict(), os.path.join(args.save_dir, f'{normal_class}/best_model.pdopt'))


def main():
    args = parser.parse_args()
    config = get_config(args.config)
    config['normal_class'] = args.normal_class
    train(args, config)


if __name__ == '__main__':
    main()
