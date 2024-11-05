import torch
import numpy as np
import copy
import torch.nn as nn
from scipy.stats import chi2, binom
from scipy.stats.contingency import crosstab, chi2_contingency

class LimeNet():
    def __init__(self, wm_length, max_length, accelerator, mask_token, wm, lam=0.0, wm_bs=1, epsilon=1e-2, max_mask_token_size=None, num_augu=None):
        self.wm_length = wm_length
        self.max_length = max_length
        self.accelerator = accelerator
        self.lam = lam
        self.block_size = int(self.max_length / self.wm_length)
        self.mask_token = mask_token
        self.wm = wm
        self.torch_wm = torch.Tensor(wm).to(self.accelerator.device).unsqueeze(-1)
        self.epsilon = epsilon
        self.wm_bs = wm_bs
        self.max_mask_token_size = max_mask_token_size
        self.num_augu = num_augu
        if self.max_mask_token_size is not None and self.block_size > self.max_mask_token_size:
            self.block_size = self.max_mask_token_size

        # generate mask
        self.flatten_masks = []
        if self.num_augu is None:
            for i in range(self.wm_length):
                flatten_mask = np.ones(self.wm_length)
                flatten_mask[i] = 0
                self.flatten_masks.append(flatten_mask)
        else:
            for i in range(self.num_augu):
                flatten_mask = np.random.choice([0, 1], self.wm_length)
                self.flatten_masks.append(flatten_mask)
        self.flatten_masks = torch.from_numpy(np.array(self.flatten_masks)).to(self.accelerator.device)
        flatten_mask = self.flatten_masks.type(torch.float32)
        self.weight_matrix = torch.mm(flatten_mask.T, flatten_mask)
        self.weight_matrix = self.weight_matrix + torch.eye(self.wm_length).to(self.accelerator.device) * self.lam
        self.weight_matrix = self.weight_matrix.inverse()
        self.weight_matrix = torch.mm(self.weight_matrix, flatten_mask.T)
        self.weight_matrix = self.weight_matrix.to(self.accelerator.device)
        self.flatten_masks.to(self.accelerator.device)

    def explain(self, model, data):
        output = self._get_prediction(model, data)
        logits = output.logits
        losses = []
        for idx in range(logits.shape[0]):
            probs = 0.0
            for j in range(logits.shape[1]):
                probs += logits[idx, j, data["labels"][j]]
            loss = probs / logits.shape[1]
            losses.append(loss)
        preds = torch.stack(losses).unsqueeze(-1)
        weights = torch.mm(self.weight_matrix, preds)
        extracted_wm = weights.cpu().detach().numpy().squeeze()
        extracted_wm[extracted_wm >= 0] = 1
        extracted_wm[extracted_wm < 0] = -1
        return extracted_wm
    
    def embed_watermark(self, model, data, alpha):
        output = self._get_prediction(model, data)
        logits = output.logits
        probs = []
        for idx in range(logits.shape[1]):
            prob = logits[:, idx, data["labels"][idx]].squeeze()
            probs.append(prob)
        losses = torch.mean(torch.stack(probs), axis=0)
        preds = losses.unsqueeze(-1)
        weights = torch.mm(self.weight_matrix, preds)
        loss_func = HingeLikeLoss(self.epsilon)
        wm_loss = alpha * loss_func(weights, self.torch_wm)
        self.accelerator.backward(wm_loss)
        return weights, wm_loss
    
    def embed_watermark_with_alm(self, model, data, rho, lam):
        output = self._get_prediction(model, data)
        logits = output.logits
        probs = []
        for idx in range(logits.shape[1]):
            prob = logits[:, idx, data["labels"][idx]].squeeze()
            probs.append(prob)
        losses = torch.mean(torch.stack(probs), axis=0)
        preds = losses.unsqueeze(-1)
        weights = torch.mm(self.weight_matrix, preds)
        loss_func = HingeLikeLoss(self.epsilon)
        wm_single_loss = loss_func(weights, self.torch_wm)
        wm_loss = rho / 2 * torch.sum(wm_single_loss ** 2) - lam * wm_single_loss
        self.accelerator.backward(wm_loss)
        return weights, wm_loss, wm_single_loss
    
    
    def _get_prediction(self, model, data):
        input_ids = []
        attention_mask = []
        labels = []
        if self.num_augu is None:
            for idx in range(self.wm_length):
                masked_text = np.array(copy.deepcopy(data["input_ids"]))
                masked_text[idx * self.block_size: (idx + 1) * self.block_size] = self.mask_token
                masked_text = torch.from_numpy(masked_text).to(self.accelerator.device)
                attention_m = torch.Tensor(copy.deepcopy(data["attention_mask"])).type(torch.int64).to(self.accelerator.device)
                label = torch.Tensor(copy.deepcopy(data["labels"])).type(torch.int64).to(self.accelerator.device)
                input_ids.append(masked_text)
                attention_mask.append(attention_m)
                labels.append(label)
        else:
            for i in range(self.num_augu):
                masked_text = np.array(copy.deepcopy(data["input_ids"]))
                for idx in range(self.wm_length):
                    if self.flatten_masks[i][idx] == 0:
                        masked_text[idx * self.block_size: (idx + 1) * self.block_size] = self.mask_token
                masked_text = torch.from_numpy(masked_text).to(self.accelerator.device)
                attention_m = torch.Tensor(copy.deepcopy(data["attention_mask"])).type(torch.int64).to(self.accelerator.device)
                label = torch.Tensor(copy.deepcopy(data["labels"])).type(torch.int64).to(self.accelerator.device)
                input_ids.append(masked_text)
                attention_mask.append(attention_m)
                labels.append(label)
        masked_data = {
            "input_ids": torch.stack(input_ids).to(self.accelerator.device),
            "attention_mask": torch.stack(attention_mask).to(self.accelerator.device),
            "labels": torch.stack(labels).to(self.accelerator.device)
        }
        outputs = model(**masked_data)
        return outputs
        

def init_watermark(wm_length):
    wm = np.random.choice([-1, 1], wm_length)
    return wm


def bit_error_rate(source, target):
    return np.average(source != target)


class HingeLikeLoss(nn.Module):
    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, results, labels):
        loss = self.epsilon - results * labels
        loss = torch.sum(torch.relu(loss))
        return loss
    

def evaluate_watermark(weights, target):
    # evaluate Bit Error Rate
    ber = bit_error_rate(weights, target)
    zero_target = target.copy()
    zero_target[zero_target == -1] = 0

    # calculate the p-value of chi2 test
    zero_weights = weights.copy().astype(np.int32)
    zero_weights[zero_weights == -1] = 0
    options = [1, 0]
    cross_table = crosstab(zero_weights, zero_target, levels=(options, options))
    stats = chi2_contingency(cross_table)
    
    return 1 - ber, stats
