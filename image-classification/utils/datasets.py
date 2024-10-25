# -*- coding: UTF-8 -*-

from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, MNIST, ImageFolder, ImageNet


def get_full_dataset(dataset_name, img_size=(32, 32)):
    if dataset_name == 'mnist':
        train_dataset = MNIST('./data/mnist/', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Resize(img_size),
                                  transforms.RandomHorizontalFlip(),
                              ]))
        test_dataset = MNIST('./data/mnist/', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Resize(img_size),
                             ]))
        num_classes = 10
        num_channels = 1
    elif dataset_name == 'cifar10':
        train_dataset = CIFAR10('./data/cifar10/', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize(img_size),
                                    transforms.Pad(4, padding_mode="reflect"),
                                    transforms.RandomCrop(img_size),
                                    transforms.RandomHorizontalFlip(),
                                ]))
        test_dataset = CIFAR10('./data/cifar10/', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Resize(img_size),
                               ]))
        num_classes = 10
        num_channels = 3
    elif dataset_name == "imagenet":
        train_dataset = ImageNet("./data/imagenet/", split="train",
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Resize(img_size),
                                     transforms.Pad(32, padding_mode="reflect"),
                                     transforms.RandomCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                 ]))
        test_dataset = ImageNet("./data/Imagenet/", split="val",
                                transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Resize(img_size),
                               ]))
        num_classes = 1000
        num_channels = 3
    elif dataset_name == "imagenet100":
        train_dataset = ImageFolder("./data/benign_100/train/", 
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Pad(32, padding_mode="reflect"),
                                        transforms.RandomCrop(img_size),
                                        transforms.RandomHorizontalFlip(),
                                    ]))
        test_dataset = ImageFolder("./data/benign_100/val/", 
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Resize(img_size),
                                   ]))
        num_classes = 100
        num_channels = 3
    else:
        exit("Unknown Dataset")
    return train_dataset, test_dataset, num_classes, num_channels
