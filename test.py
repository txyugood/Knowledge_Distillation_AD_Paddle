from argparse import ArgumentParser
from utils.utils import get_config, load_pretrained_model
from test_functions import detection_test, localization_test
from network import get_networks
from dataloader import load_data, load_localization_data

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")
parser.add_argument('--dataset_root', type=str, default=None)
parser.add_argument('--model_path', type=str, default=None)


def main():
    args = parser.parse_args()
    config = get_config(args.config)
    vgg, model = get_networks(config)

    if args.model_path is not None:
        load_pretrained_model(model, args.model_path)

    # Localization test
    if config['localization_test']:
        test_dataloader, ground_truth = load_localization_data(args, config)
        roc_auc = localization_test(model=model, vgg=vgg, test_dataloader=test_dataloader, ground_truth=ground_truth,
                                    config=config)

    # Detection test
    else:
        _, test_dataloader = load_data(args, config)
        roc_auc = detection_test(model=model, vgg=vgg, test_dataloader=test_dataloader, config=config)
    last_checkpoint = config['last_checkpoint']
    print("RocAUC after {} epoch:".format(last_checkpoint), roc_auc)


if __name__ == '__main__':
    main()
