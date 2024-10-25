# -*- coding: UTF-8 -*-
import os.path
import time
import numpy as np
import torch
import torch.nn as nn
import random
import math
import json
import logging
import copy
import datetime
from tqdm import tqdm
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel as DDP
from torchvision.transforms import ToPILImage, ToTensor
import cv2 as cv

from utils.datasets import get_full_dataset
from utils.models import get_model
from utils.test import test_img
from utils.train import get_optim
from utils.utils import load_args

from watermark.watermark import *
from watermark.lime import LimeNet



def train(args):
    args.save_path = os.path.join(
        args.save_dir, 
        args.wm_mode, 
        args.dataset, 
        args.model, 
        args.mode, 
        datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    )
    # create save dir
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # save args
    with open(os.path.join(args.save_path, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True

    # set log
    log_path = os.path.join(args.save_path, 'log.log')
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path)
        ]
    )

    # set device
    gpus = [int(gpu) for gpu in args.gpus.split(",")]
    args.device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() and gpus[0] != -1 else 'cpu')

    # load dataset
    logger.info("Load Dataset:{}".format(args.dataset))
    train_dataset, test_dataset, num_classes, num_channels = get_full_dataset(args.dataset, img_size=(args.image_size, args.image_size))
    args.num_classes = num_classes
    args.num_channels = num_channels
    logger.info("Train Dataset Samples: {}, Test Dataset Samples: {}".format(len(train_dataset), len(test_dataset)))
    
    # load model
    model = get_model(args)
    if args.pre_train_path is not None:
        model.load_state_dict(torch.load(args.pre_train_path, map_location="cpu"))
    model = DDP(model, device_ids=gpus)
    model.to(args.device)

    # if load the pretrained model, test the initial preformance
    # if args.pre_train_path is not None:
    acc_val, _ = test_img(model, test_dataset, args)
    logger.info("Initial Accuracy: {:.3f}".format(acc_val))

    logger.info("Model is ready.")
    basic_loss = nn.CrossEntropyLoss()

    acc_best = None
    es_count = 0
    # --------------------------------mode 0: train without watermark----------------------------------
    if args.wm_mode == "nowm":
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=16, pin_memory=True)
        optim = get_optim(model.parameters(), args.optim, args.lr, args.momentum, args.wd)
        schedule = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.1)
        for epoch in tqdm(range(args.start_epochs, args.start_epochs + args.epochs)):
            batch_loss = []
            model.train()
            idx = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                optim.zero_grad()
                x, y = x.to(args.device), y.to(args.device)
                preds = model(x)
                loss = basic_loss(preds, y)
                loss.backward()
                optim.step()
                logger.info("Epoch {} Step {}: LR: {}, loss: {:.4f}".format(epoch, batch_idx, optim.state_dict()["param_groups"][0]["lr"], loss.item()))
                batch_loss.append(loss.item())
            epoch_loss = np.mean(batch_loss)
            logger.info("---------------End of Epoch {}---------------".format(epoch))
            logger.info("Epoch {} loss:{:.4f}".format(epoch, epoch_loss))
            if (epoch + 1) % args.eval_rounds == 0:
                acc_val, loss_val = test_img(model, test_dataset, args)
                logger.info("Epoch {} val loss:{:.4f}, val acc:{:.3f}".format(epoch, loss_val, acc_val))
                if acc_best is None or acc_best < acc_val:
                    acc_best = acc_val
                    if args.save_model:
                        torch.save(model.module.state_dict(), os.path.join(args.save_path, "model_best.pth"))
                    es_count = 0
                else:
                    es_count += 1
                    if es_count >= args.stopping_rounds:
                        break
            schedule.step()
    # -----------------------------------------------mode 1: embed watermark with fixed trigger set-----------------------------------------------
    elif args.wm_mode == "wm":
        # prepare watermark trigger set
        triggered_data, triggered_labels = get_fixed_trigger(train_dataset, trigger_size=args.trigger_size,
                                                             image_size=args.image_size, num_channels=args.num_channels,
                                                             pattern_path=args.pattern_path,
                                                             base_trigger=args.base_trigger, num_classes=args.num_classes,
                                                             mix_type=args.mix_type)
        perturbed_image_path = os.path.join(args.save_path, 'perturbed_images/')
        if not os.path.exists(perturbed_image_path):
            os.makedirs(perturbed_image_path)
        explained_image_path = os.path.join(args.save_path, "explained_images/")
        if not os.path.exists(explained_image_path):
            os.makedirs(explained_image_path)
        # save the trigger set
        for idx in range(args.trigger_size):
            data = triggered_data[idx]
            # save perturbed images
            perturbed_image = torch.clamp(data, 0, 1)
            perturbed_image = ToPILImage()(perturbed_image)
            perturbed_image.save(os.path.join(perturbed_image_path, "perturbed_image_{}.png".format(idx)))
        np.savetxt(os.path.join(perturbed_image_path, "labels.txt"), triggered_labels.numpy(), fmt="%d", delimiter=",")
        triggered_data, triggered_labels = triggered_data.to(args.device), triggered_labels.long().to(args.device)
        optim = get_optim(model.parameters(), args.optim, args.lr, args.momentum, args.wd)
        schedule = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
        # construct LimeNet
        lime_model = LimeNet(args.num_mask, args.wm_length, args.image_size, args.num_channels, 
                             args.device, args.lam)
        # get target image
        target = get_target_images(args.target_path, args.wm_length, args.save_path, args.device)
        wm_loss_fn = get_watermark_loss(args)
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=16, pin_memory=True)
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1 or classname.find('LayerNorm') != -1:
                m.eval()

        for epoch in tqdm(range(args.start_epochs, args.start_epochs + args.epochs)):
            batch_loss = []
            model.train()
            model.apply(set_bn_eval)
            for batch_idx, (x, y) in enumerate(train_loader):
                optim.zero_grad()
                x, y = x.to(args.device), y.to(args.device)
                x, y = torch.cat([x, triggered_data], dim=0), torch.cat([y, triggered_labels])
                preds = model(x)
                benign_loss = basic_loss(preds, y)
                benign_loss.backward()
                # embed the watermark
                for i in range(len(triggered_labels)):
                    weights = lime_model.explain(model, triggered_data[i], triggered_labels[i], no_logits=False)
                    wm_loss = args.alpha1 * wm_loss_fn(weights.squeeze(), target.float().to(args.device)) / len(triggered_labels)
                    wm_loss.backward()
                optim.step()
                batch_loss.append([float(benign_loss.item()), float(wm_loss.item())])

            schedule.step()
            batch_loss = np.array(batch_loss)
            epoch_loss = np.mean(batch_loss, axis=0)
            logger.info("---------------End of Epoch {}---------------".format(epoch))
            logger.info("Epoch {}, LR: {}, benign loss:{}, wm loss:{}".format(epoch, optim.state_dict()["param_groups"][0]["lr"], *epoch_loss))
            if (epoch + 1) % args.eval_rounds == 0:
                acc_val, loss_val = test_img(model, test_dataset, args)
                logger.info("Epoch {} val loss:{:.4f}, val acc:{:.3f}".format(epoch, loss_val, acc_val))
                if acc_best is None or acc_best < acc_val:
                    acc_best = acc_val
                    if args.save_model:
                        torch.save(model.module.state_dict(), os.path.join(args.save_path, "model_best.pth"))
                    es_count = 0
                else:
                    es_count += 1
                    if es_count >= args.stopping_rounds:
                        break
            wsrs = []
            chi2_corrs = []
            chi2_p_values = []
            trigger_correct = 0.0
            for i in range(len(triggered_labels)):
                data = triggered_data[i]
                label = triggered_labels[i]
                pred = model(data.unsqueeze(0))
                _, y_pred = torch.max(pred.data, 1)
                if y_pred[0] == label:
                    trigger_correct += 1
                with torch.no_grad():
                    weights = lime_model.explain(model, data, label, no_logits=args.no_logits)
                wsr, stats = evaluate_watermark(weights, target)
                wsrs.append(wsr)
                chi2_corrs.append(stats.statistic)
                chi2_p_values.append(stats.pvalue)
                explained_image = lime_model.explain_image(model, data, label, no_logits=args.no_logits)
                explained_image = explained_image.repeat(3, axis=-1)
                explained_image = Image.fromarray(np.uint8(explained_image))
                explained_image.save(os.path.join(explained_image_path, "explained_images_{}.png".format(i)))
            logger.info("Trigger Accuracy:{}".format(trigger_correct / args.trigger_size))
            logger.info("Average WSR:{}".format(np.average(wsrs)))
            logger.info("chi2 Test: Average Corr:{}, Average P-value:{}".format(np.average(chi2_corrs), np.average(chi2_p_values)))

    if args.save_model:
        torch.save(model.module.state_dict(), os.path.join(args.save_path, "model_last_epochs_" + str(epoch) + ".pth"))
        model.module.load_state_dict(torch.load(os.path.join(args.save_path, "model_best.pth")))
        acc_test, _ = test_img(model, test_dataset, args)
        logger.info("Best Testing Accuracy:{:.2f}".format(acc_test))


def test(args):
    args.save_path = os.path.join(
        args.save_dir, 
        args.wm_mode, 
        args.dataset, 
        args.model, 
        args.mode, 
        datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    )
    # create save dir
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # save args
    with open(os.path.join(args.save_path, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True

    # set log
    log_path = os.path.join(args.save_path, 'log.log')
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m-%d-%Y-%H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path)
        ]
    )

    explained_image_path = os.path.join(args.save_path, "explained_images/")
    if not os.path.exists(explained_image_path):
        os.makedirs(explained_image_path)

    # set device
    gpus = [int(gpu) for gpu in args.gpus.split(",")]
    args.device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() and gpus[0] != -1 else 'cpu')

    # load dataset
    logger.info("Load Dataset:{}".format(args.dataset))
    train_dataset, test_dataset, num_classes, num_channels = get_full_dataset(args.dataset, img_size=(args.image_size, args.image_size))
    args.num_classes = num_classes
    args.num_channels = num_channels
    
    # load model
    model = get_model(args)
    if args.pre_train_path is not None:
        model.load_state_dict(torch.load(args.pre_train_path))
    model = DDP(model, device_ids=gpus)
    model.to(args.device)

    acc_val, _ = test_img(model, test_dataset, args)
    logger.info("Primitive Accuracy: {:.3f}".format(acc_val))

    # load testing images
    testing_images = []
    for idx in range(args.trigger_size):
        image_path = os.path.join(args.test_image_path, "perturbed_image_{}.png".format(idx))
        im = Image.open(image_path)
        if args.num_channels == 1:
            im = im.convert("L")
        im = ToTensor()(im)
        testing_images.append(im)
    
    # load labels
    with open(os.path.join(args.test_image_path, "labels.txt")) as f:
        for line in f:
            labels = [int(label) for label in line.split(",")]
            break
    labels = torch.Tensor(labels)

    # construct lime model
    lime_model = LimeNet(args.num_mask, args.wm_length, args.image_size, args.num_channels, 
                             args.device, args.lam)
    # get target image
    target = get_target_images(args.target_path, args.wm_length, args.save_path, args.device)
    wsrs = []
    chi2_corrs = []
    chi2_p_values = []
    for idx in range(len(testing_images)):
        data = testing_images[idx]
        label = labels[idx]
        data = data.to(args.device)
        label = label.to(args.device)
        with torch.no_grad():
            weights = lime_model.explain(model, data, label, no_logits=args.no_logits)
        wsr, chi2_stats = evaluate_watermark(weights, target)
        wsrs.append(wsr)
        chi2_corrs.append(chi2_stats.statistic)
        chi2_p_values.append(chi2_stats.pvalue)
        explained_image = lime_model.explain_image(model, data, label, no_logits=args.no_logits)
        explained_image = explained_image.repeat(3, axis=-1)
        explained_image = Image.fromarray(np.uint8(explained_image))
        explained_image.save(os.path.join(explained_image_path, "explained_images_{}.png".format(idx)))
    logger.info("Average WSR:{}".format(np.average(wsrs)))
    logger.info("Chi2 Test: Average Corr:{}, Average P-value:{}".format(np.average(chi2_corrs), np.average(chi2_p_values)))



if __name__ == '__main__':
    args = load_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    
        
