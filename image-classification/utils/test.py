# -*- coding: UTF-8 -*-
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F


def test_img(model, datatest, args):
    model.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.test_bs)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        with torch.no_grad():
            log_probs = model(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        _, y_pred = torch.max(log_probs.data, 1)
        correct += (y_pred == target).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss


def test_trigger(model, trigger_data, trigger_labels, args):
    model.eval()
    data, label = trigger_data.to(args.device), trigger_labels.to(args.device)
    with torch.no_grad():
        probs = model(trigger_data)
    test_loss = F.cross_entropy(probs, label, reduction='mean').item()
    # get the index of the max log-probability
    _, y_pred = torch.max(probs.data, 1)
    accuracy = (y_pred == label).sum().item() / args.trigger_size
    return accuracy, test_loss
