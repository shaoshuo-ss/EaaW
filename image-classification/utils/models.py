# -*- coding: UTF-8 -*-

from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, swin_t, vit_b_16, inception_v3, densenet121, resnet101


def get_resnet18(args):
    model = resnet18(weights=None, num_classes=args.num_classes)
    if args.dataset == "cifar10":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    return model


def get_resnet34(args):
    model = resnet34(weights=None, num_classes=args.num_classes)
    return model


def get_resnet50(args):
    model = resnet50(weights=None, num_classes=args.num_classes)
    return model

def get_resnet101(args):
    model = resnet101(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    return model


def get_swinvit(args):
    model = swin_t(weights=None, num_classes=args.num_classes)
    return model


def get_vit(args):
    model = vit_b_16(weights=None, num_classes=args.num_classes)
    return model

def get_model(args):
    if args.model == 'ResNet18':
        return get_resnet18(args)
    elif args.model == 'ResNet50':
        return get_resnet50(args)
    elif args.model == 'ResNet34':
        return get_resnet34(args)
    elif args.model == 'ResNet101':
        return get_resnet101(args)
    elif args.model == "Swin_ViT":
        return get_swinvit(args)
    elif args.model == "InceptionV3":
        return inception_v3(weights=None, num_classes=args.num_classes)
    elif args.model == "DenseNet121":
        return densenet121(weights=None, num_classes=args.num_classes)
    else:
        exit("Unknown Model!")
