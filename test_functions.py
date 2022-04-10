from copy import deepcopy

import paddle
from paddle import nn
from sklearn.metrics import roc_curve, auc
from scipy.ndimage.filters import gaussian_filter
import numpy as np

from utils.utils import morphological_process, convert_to_grayscale, max_regarding_to_abs


def detection_test(model, vgg, test_dataloader, config):
    normal_class = config["normal_class"]
    lamda = config['lamda']
    dataset_name = config['dataset_name']
    direction_only = config['direction_loss_only']

    if dataset_name != "mvtec":
        target_class = normal_class
    else:
        mvtec_good_dict = {'bottle': 3, 'cable': 5, 'capsule': 2, 'carpet': 2,
                           'grid': 3, 'hazelnut': 2, 'leather': 4, 'metal_nut': 3, 'pill': 5,
                           'screw': 0, 'tile': 2, 'toothbrush': 1, 'transistor': 3, 'wood': 2,
                           'zipper': 4
                           }
        target_class = mvtec_good_dict[normal_class]

    similarity_loss = paddle.nn.CosineSimilarity()
    label_score = []
    model.eval()
    for data in test_dataloader:
        X, Y = data
        if X.shape[1] == 1:
            X = X.repeat(1, 3, 1, 1)
        output_pred = model.forward(X)
        output_real = vgg(X)
        y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[9], output_pred[12]
        y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]

        if direction_only:
            loss_1 = 1 - similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1))
            loss_2 = 1 - similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1))
            loss_3 = 1 - similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1))
            total_loss = loss_1 + loss_2 + loss_3
        else:
            abs_loss_1 = paddle.mean((y_pred_1 - y_1) ** 2, axis=(1, 2, 3))
            loss_1 = 1 - similarity_loss(y_pred_1.reshape([y_pred_1.shape[0], -1]), y_1.reshape([y_1.shape[0], -1]))
            abs_loss_2 = paddle.mean((y_pred_2 - y_2) ** 2, axis=(1, 2, 3))
            loss_2 = 1 - similarity_loss(y_pred_2.reshape([y_pred_2.shape[0], -1]), y_2.reshape([y_2.shape[0], -1]))
            abs_loss_3 = paddle.mean((y_pred_3 - y_3) ** 2, axis=(1, 2, 3))
            loss_3 = 1 - similarity_loss(y_pred_3.reshape([y_pred_3.shape[0], -1]), y_3.reshape([y_3.shape[0], -1]))
            total_loss = loss_1 + loss_2 + loss_3 + lamda * (abs_loss_1 + abs_loss_2 + abs_loss_3)

        label_score += list(zip(Y.detach().numpy().tolist(), total_loss.detach().numpy().tolist()))

    labels, scores = zip(*label_score)
    labels = np.array(labels)
    indx1 = labels == target_class
    indx2 = labels != target_class
    labels[indx1] = 1
    labels[indx2] = 0
    scores = np.array(scores)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
    roc_auc = auc(fpr, tpr)
    roc_auc = round(roc_auc, 4)
    return roc_auc


def localization_test(model, vgg, test_dataloader, ground_truth, config):
    localization_method = config['localization_method']
    if localization_method == 'gradients':
        grad = gradients_localization(model, vgg, test_dataloader, config)

    return compute_localization_auc(grad, ground_truth)


def grad_calc(inputs, model, vgg, config):
    inputs.stop_gradient = False
    temp = paddle.zeros(inputs.shape)
    lamda = config['lamda']
    criterion = nn.MSELoss()
    similarity_loss = paddle.nn.CosineSimilarity()

    for i in range(inputs.shape[0]):
        output_pred = model.forward(inputs[i].unsqueeze(0), target_layer=14)
        output_real = vgg(inputs[i].unsqueeze(0))
        y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[9], output_pred[12]
        y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]
        abs_loss_1 = criterion(y_pred_1, y_1)
        loss_1 = paddle.mean(1 - similarity_loss(y_pred_1.reshape([y_pred_1.shape[0], -1]), y_1.reshape([y_1.shape[0], -1])))
        abs_loss_2 = criterion(y_pred_2, y_2)
        loss_2 = paddle.mean(1 - similarity_loss(y_pred_2.reshape([y_pred_2.shape[0], -1]), y_2.reshape([y_2.shape[0], -1])))
        abs_loss_3 = criterion(y_pred_3, y_3)
        loss_3 = paddle.mean(1 - similarity_loss(y_pred_3.reshape([y_pred_3.shape[0], -1]), y_3.reshape([y_3.shape[0], -1])))
        total_loss = loss_1 + loss_2 + loss_3 + lamda * (abs_loss_1 + abs_loss_2 + abs_loss_3)
        model.clear_gradients()
        total_loss.backward()

        temp[i] = inputs.grad[i]

    return temp


def gradients_localization(model, vgg, test_dataloader, config):
    model.eval()
    print("Vanilla Backpropagation:")
    temp = None
    for data in test_dataloader:
        X, Y = data
        grad = grad_calc(X, model, vgg, config)
        temp = np.zeros((grad.shape[0], grad.shape[2], grad.shape[3]))
        for i in range(grad.shape[0]):
            grad_temp = convert_to_grayscale(grad[i].cpu().numpy())
            grad_temp = grad_temp.squeeze(0)
            grad_temp = gaussian_filter(grad_temp, sigma=4)
            temp[i] = grad_temp
    return temp


def compute_localization_auc(grad, x_ground):
    tpr = []
    fpr = []
    x_ground_comp = np.mean(x_ground, axis=3)

    thresholds = [0.001 * i for i in range(1000)]

    for threshold in thresholds:
        grad_t = 1.0 * (grad >= threshold)
        grad_t = morphological_process(grad_t)
        tp_map = np.multiply(grad_t, x_ground_comp)
        tpr.append(np.sum(tp_map) / np.sum(x_ground_comp))

        inv_x_ground = 1 - x_ground_comp
        fp_map = np.multiply(grad_t, inv_x_ground)
        tn_map = np.multiply(1 - grad_t, 1 - x_ground_comp)
        fpr.append(np.sum(fp_map) / (np.sum(fp_map) + np.sum(tn_map)))

    return auc(fpr, tpr)
