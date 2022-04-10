from argparse import ArgumentParser
from utils.utils import get_config, load_pretrained_model
from test_functions import detection_test, localization_test
from network import get_networks
from dataloader import load_data, load_localization_data

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")
parser.add_argument('--dataset_root', type=str, default=None)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--normal_class', type=str, default='capsule')



def main():
    args = parser.parse_args()
    config = get_config(args.config)
    config['normal_class'] = args.normal_class
    vgg, model = get_networks(config)

    if args.model_path is not None:
        load_pretrained_model(model, args.model_path)
    dataset_name = config['dataset_name']

    # Localization test
    test_dataloader, ground_truth = load_localization_data(args, config)
    roc_auc = localization_test(model=model, vgg=vgg, test_dataloader=test_dataloader, ground_truth=ground_truth,
                                config=config)
    print(f"{dataset_name}: {args.normal_class} class localization test RocAUC: {roc_auc}")

    # Detection test
    _, test_dataloader = load_data(args, config)
    roc_auc = detection_test(model=model, vgg=vgg, test_dataloader=test_dataloader, config=config)

    print(f"{dataset_name}: {args.normal_class} class detection test RocAUC: {roc_auc}")


if __name__ == '__main__':
    main()
