import paddle.nn as nn

from vgg import vgg16
from param_init import xavier_uniform_


class VGG(nn.Layer):
    '''
    VGG model
    '''

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

        # placeholder for the gradients
        self.gradients = None
        self.activation = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, target_layer=11):
        result = []
        for i in range(len(nn.LayerList(self.features))):
            x = self.features[i](x)
            if i == target_layer:
                self.activation = x
                h = x.register_hook(self.activations_hook)
            if i == 2 or i == 5 or i == 8 or i == 11 or i == 14 or i == 17 or i == 20 or i == 23 or i == 26 or i == 29 or i == 32 or i == 35 or i == 38:
                result.append(x)

        return result

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.activation


def make_layers(cfg, use_bias, batch_norm=False):
    layers = []
    in_channels = 3
    outputs = []
    for i in range(len(cfg)):
        if cfg[i] == 'O':
            outputs.append(nn.Sequential(*layers))
        elif cfg[i] == 'M':
            layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2D(in_channels, cfg[i], kernel_size=3, padding=1, bias_attr=use_bias)
            xavier_uniform_(conv2d.weight)
            if batch_norm and cfg[i + 1] != 'M':
                layers += [conv2d, nn.BatchNorm2D(cfg[i]), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = cfg[i]
    return nn.Sequential(*layers)


def make_arch(idx, cfg, use_bias, batch_norm=False):
    return VGG(make_layers(cfg[idx], use_bias, batch_norm=batch_norm))


class Vgg16(nn.Layer):
    def __init__(self, pretrained):
        super(Vgg16, self).__init__()

        features = list(vgg16(pretrained).features)

        self.features = nn.LayerList(features)
        self.features.eval()
        self.output = []

    def forward(self, x):
        output = []
        for i in range(31):
            x = self.features[i](x)
            if i == 1 or i == 4 or i == 6 or i == 9 or i == 11 or i == 13 or i == 16 or i == 18 or i == 20 or i == 23 or i == 25 or i == 27 or i == 30:
                output.append(x)
        return output


def get_networks(config, load_checkpoint=False):
    equal_network_size = config['equal_network_size']
    pretrain = config['pretrain']
    experiment_name = config['experiment_name']
    dataset_name = config['dataset_name']
    normal_class = config['normal_class']
    use_bias = config['use_bias']
    cfg = {
        'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'B': [16, 16, 'M', 16, 128, 'M', 16, 16, 256, 'M', 16, 16, 512, 'M', 16, 16, 512, 'M'],
    }

    if equal_network_size:
        config_type = 'A'
    else:
        config_type = 'B'

    vgg = Vgg16('./vgg16.pdparams')
    model = make_arch(config_type, cfg, use_bias, True)

    for j, item in enumerate(nn.LayerList(model.features)):
        print('layer : {} {}'.format(j, item))

    return vgg, model
