# -*- coding: UTF-8 -*-


import numpy as np
import torch
import random
import math


class LimeNet():
    
    def __init__(self, num_mask, wm_length, image_size, num_channels, device, lam=0.0):
        if num_mask is not None:
            self.num_mask = num_mask
        else:
            self.num_mask = wm_length
        self.wm_length = wm_length
        self.image_size = image_size
        self.num_channels = num_channels
        self.block_size = int(image_size / math.sqrt(wm_length))
        self.lam = lam
        self.device = device
        # randomly generate mask

        self.masks = []
        self.flatten_mask = []
        if num_mask is None:
            for i in range(self.wm_length):
                flatten_mask = np.ones(self.wm_length)
                flatten_mask[i] = 0
                self.flatten_mask.append(flatten_mask)
                mask_matrix = np.ones((num_channels, image_size, image_size))
                row_num = int(image_size / self.block_size)
                row_count = int(i // row_num)
                col_count = int(i % row_num)
                mask_matrix[:, row_count * self.block_size: (row_count + 1) * self.block_size,
                            col_count * self.block_size: (col_count + 1) * self.block_size] = 0
                self.masks.append(mask_matrix)
        else:
            self.flatten_mask = np.random.choice([0, 1], self.wm_length, p=[0.5, 0.5]).reshape(1, self.wm_length)
            while self.flatten_mask.shape[0] < self.num_mask:
                flatten_mask = np.random.choice([0, 1], self.wm_length, p=[0.5, 0.5])
                self.flatten_mask = np.r_[self.flatten_mask, [flatten_mask]]
            for i in range(self.num_mask):
                flatten_mask = self.flatten_mask[i]
                mask_matrix = np.ones((num_channels, image_size, image_size))
                row_num = int(image_size / self.block_size)
                row_count = 0
                col_count = 0
                for j in range(self.wm_length):
                    if flatten_mask[j] == 0:
                        mask_matrix[:, row_count * self.block_size: (row_count + 1) * self.block_size,
                            col_count * self.block_size: (col_count + 1) * self.block_size] = 0
                    row_count += 1
                    if row_count == row_num:
                        row_count = 0
                        col_count += 1
                self.masks.append(mask_matrix)
        self.masks = torch.from_numpy(np.array(self.masks)).to(device)
        self.flatten_mask = torch.from_numpy(np.array(self.flatten_mask)).to(device)
        flatten_mask = self.flatten_mask.type(torch.float32)
        self.weight_matrix = torch.mm(flatten_mask.T, flatten_mask)
        self.weight_matrix = self.weight_matrix + torch.eye(self.wm_length).to(device) * self.lam
        self.weight_matrix = self.weight_matrix.inverse()
        self.weight_matrix = torch.mm(self.weight_matrix, flatten_mask.T)
        self.weight_matrix = self.weight_matrix.to(self.device)

    
    def explain(self, model, image, label, no_logits=False):
        masked_images = self.masks * image
        masked_images = masked_images.float().to(self.device)
        
        if not no_logits:
            predictions = model(masked_images)[:, int(label.item())].unsqueeze(-1)
        elif no_logits:
            preds = model(masked_images)
            predictions = []
            # print(preds.shape)
            for i in range(preds.shape[0]):
                # print(preds.shape)
                pred_class = torch.argmax(preds[i])
                # print(pred_class.item())
                # print(label.item())
                if int(pred_class.item()) != int(label.item()):
                    predictions.append(0)
                else:
                    predictions.append(1)
            print(np.count_nonzero(predictions))
            predictions = torch.Tensor(predictions).unsqueeze(-1)
            predictions = predictions.to(self.device)
        
        weight = torch.mm(self.weight_matrix, predictions)
        return weight
    
    def explain_image(self, model, image, label, no_logits=False):
        weight = self.explain(model, image, label, no_logits)
        weight = weight.reshape((int(math.sqrt(self.wm_length)), int(math.sqrt(self.wm_length)), 1)).cpu().detach().numpy()
        weight[weight > 0] = 255
        weight[weight <= 0] = 0
        return weight


def dec2bin(num, length):
    mid = []
    while True:
        if num == 0:
            break
        num, rem = divmod(num, 2)
        if int(rem) == 0:
            mid.append(0)
        else:
            mid.append(1)
        # mid.append(int(rem))
    while len(mid) < length:
        mid.insert(0, 0)
    return mid

