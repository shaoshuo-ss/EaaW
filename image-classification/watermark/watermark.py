# -*- coding: UTF-8 -*-

import os.path

import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import random_split
from piqa import SSIM
from scipy.stats import chi2, binom
from scipy.stats.contingency import crosstab, chi2_contingency
from torchvision.datasets import ImageFolder


def select_triggered_data(train_dataset, trigger_size, wrong_label=False):
    triggered_data = []
    triggered_labels = []
    trigger_set, _ = random_split(train_dataset, [trigger_size, len(train_dataset) - trigger_size])
    for data, label in trigger_set:
        triggered_data.append(data.clone())
        if not wrong_label:
            triggered_labels.append(label)
        else:
            if label != 0:
                triggered_labels.append(label - 1)
            else:
                triggered_labels.append(label + 1)
    return triggered_data, triggered_labels


def select_random_noise_data(trigger_size, image_size, num_channels, num_classes):
    triggered_data = []
    triggered_labels = []
    label_idx = 0
    for _ in range(trigger_size):
        data = torch.Tensor(np.random.standard_normal((num_channels, image_size, image_size)))
        label = label_idx
        triggered_data.append(data)
        triggered_labels.append(label)
        label_idx += 1
        if label_idx == num_classes:
            label_idx = 0
    return triggered_data, triggered_labels



def get_fixed_trigger(train_dataset, trigger_size, image_size=32, num_channels=3, pattern_path=None, base_trigger="image", num_classes=10, mix_type="mix", wrong_label=False):
    # load pattern
    pattern = Image.open(pattern_path)
    if num_channels == 1:
        pattern = pattern.convert("L")
    else:
        pattern = pattern.convert("RGB")
    pattern = np.array(pattern)
    pattern = cv.resize(pattern, (image_size, image_size))
    pattern = np.transpose(pattern, [2, 0, 1])
    if base_trigger == "image":
        triggered_data, triggered_labels = select_triggered_data(train_dataset, trigger_size, wrong_label=wrong_label)
    elif base_trigger == "noise":
        triggered_data, triggered_labels = select_random_noise_data(trigger_size, image_size, num_channels, num_classes)
        return torch.stack(triggered_data).type(torch.float32), torch.Tensor(triggered_labels)
    elif base_trigger == "pattern":
        pattern = pattern / 255
        pattern = np.clip(pattern, 0, 1)
        pattern = torch.from_numpy(pattern)
        triggered_data = [pattern]
        triggered_labels = np.random.choice(num_classes, 1)
        return torch.stack(triggered_data).type(torch.float32), torch.Tensor(triggered_labels)
    elif base_trigger == "mask":
        pattern = np.random.choice([0,255], (num_channels, image_size, image_size), p=[0.8, 0.2]).astype(np.float32)
        triggered_data, triggered_labels = select_triggered_data(train_dataset, trigger_size, wrong_label=wrong_label)
    pattern[pattern <= 200] = 1
    pattern[pattern > 200] = 0
    triggered_img = []
    for img in triggered_data:
        if mix_type == "mix":
            img = (img + pattern) / 2
        elif mix_type == "cover":
            img = img * (1 - pattern)
        img = np.clip(img, 0, 1)
        triggered_img.append(img)
    return torch.stack(triggered_img), torch.Tensor(triggered_labels)


def get_target_images(target_path, wm_length, save_path, device):
    target = Image.open(target_path)
    target = target.convert("L")
    target = np.array(target)
    target = cv.resize(target, (int(np.sqrt(wm_length)), int(np.sqrt(wm_length))))
    # save target image
    target_img = target.copy()
    target_img[target_img < 200] = 0
    target_img[target_img >= 200] = 255
    target_img = Image.fromarray(target_img)
    target_img.save(os.path.join(save_path, "target.png"))
    # binaryzation
    target = target.flatten()
    target = target.astype(np.int16)
    target[target < 200] = -1
    target[target >= 200] = 1
    target = torch.from_numpy(target).long().to(device)
    return target


def get_loss(wm_mode, loss, alpha1=None, epsilon=None):
    if loss == "CE":
        main_loss = torch.nn.CrossEntropyLoss()
    elif loss == "MSE":
        main_loss = torch.nn.MSELoss()
    else:
        exit("Unknown Loss")
    if wm_mode == "nowm":
        return main_loss
    elif wm_mode == "wm":
        class Target_Loss(nn.Module):
            def __init__(self, alpha):
                super().__init__()
                self.alpha = alpha

            def forward(self, prediction, y, prediction_t, y_t, w, t):
                l1 = nn.functional.cross_entropy(prediction, y, reduction='sum') + nn.functional.cross_entropy(prediction_t, y_t, reduction='sum')
                l1 = l1 / (len(y) + len(y_t))
                hl_loss = HingeLikeLoss(epsilon=epsilon)
                l2 = hl_loss(w, t)
                return  l1 + torch.multiply(self.alpha, l2)
                # return l2 * self.alpha
        return Target_Loss(alpha1)
    

class HingeLikeLoss(nn.Module):
    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, results, labels):
        loss = self.epsilon - results * labels
        loss = torch.sum(torch.relu(loss))
        return loss
    

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, results, labels):
        y = labels.clone()
        y[y < 0 ] = 0
        loss = torch.nn.functional.mse_loss(torch.sigmoid(results), y)
        return loss
    

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, results, labels):
        y = labels.clone()
        y[y < 0 ] = 0
        loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(results), y)

        return loss


class SSIMLoss(nn.Module):
    def __init__(self, device, wm_length, window_size=11, sigma=1.5, reduction="mean"):
        super().__init__()
        self.device = device
        self.shape = int(np.sqrt(wm_length))
        self.ssim = SSIM(window_size=window_size, sigma=sigma, n_channels=1, reduction=reduction).to(self.device)
    
    def forward(self, results, labels):
        img = results.reshape((1, 1, self.shape, self.shape))
        target_img = labels.reshape((1, 1, self.shape, self.shape))
        img = torch.sigmoid(img)
        target_img = torch.clamp(target_img, min=0, max=1)
        loss = 1 - self.ssim(img, target_img)
        return loss


def bit_error_rate(source, target):
    return np.average(source != target)


def get_watermark_loss(args):
    if args.wm_loss == "HingeLike":
        wm_loss_fn = HingeLikeLoss(epsilon=args.epsilon)
    elif args.wm_loss == "MSE":
        wm_loss_fn = MSELoss()
    elif args.wm_loss == "SSIM":
        wm_loss_fn = SSIMLoss(args.device, args.wm_length, window_size=args.window_size)
    elif args.wm_loss == "CE":
        wm_loss_fn = CELoss()
    return wm_loss_fn


def evaluate_watermark(weights, target):
    # evaluate Bit Error Rate
    bi_weights = weights.cpu().detach().numpy()
    bi_weights[bi_weights > 0] = 1
    bi_weights[bi_weights <= 0] = -1
    ber = bit_error_rate(bi_weights.squeeze().astype(np.int16), target.cpu().detach().numpy())
    
    # chi2-test
    zero_target = target.cpu().detach().numpy()
    zero_target[zero_target == -1] = 0
    zero_weights = weights.cpu().detach().numpy()
    zero_weights[zero_weights > 0] = 1
    zero_weights[zero_weights <= 0] = 0
    options = [1, 0]
    cross_table = crosstab(zero_weights.squeeze().astype(np.int16), zero_target, levels=(options, options))
    stats = chi2_contingency(cross_table)
    
    return 1 - ber, stats
