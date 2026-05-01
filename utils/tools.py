import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import argparse
import time
import random
from functools import wraps
from omegaconf import OmegaConf
import copy

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, train_cfg, scheduler=None, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    lradj, lr, train_epochs = train_cfg.lradj, train_cfg.lr, train_cfg.epochs
    if lradj == 'type1':
        lr_adjust = {epoch: lr * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif lradj == 'type3':
        lr_adjust = {epoch: lr if epoch < 2 else lr * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == 'constant':
        lr_adjust = {epoch: lr * 1}
    elif lradj == 'TST':
        assert scheduler is not None, "Scheduler must be provided for TST learning rate adjustment."
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, save_model=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_model = save_model
        self.best_model = None
        self.best_epoch = 0

    def __call__(self, val_loss, model, path, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            self.best_epoch = epoch

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.save_model:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        else:
            del self.best_model
            self.best_model = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def is_list(obj):
    if isinstance(obj, list):
        return True
    elif OmegaConf.is_list(obj):
        return True
    return False

def slice_list(src_list, slice_list, single_item_as_list=True):
    """
    Slice src_list into sub-lists according to slice_list.
    single_item_as_list: if True, return a list of single elements when the sub-list has only one element.
    """
    src_list = src_list if is_list(src_list) else [src_list]
    iter_list = iter(src_list)
    not_flex_len = sum([length for length in slice_list if length > 0])
    temp_flex_list = [length for length in slice_list if length < 0]
    assert len(temp_flex_list) <= 1, f"Only one flexible slice is allowed in slice_list. Found {len(temp_flex_list)} flexible slices."
    exist_flex = len(temp_flex_list) > 0
    flex_len = len(src_list) - not_flex_len if exist_flex else 0
    assert (not exist_flex and len(src_list) == not_flex_len) or (exist_flex and flex_len > 0), "The length of src_list does not match the total length of slice_list."
    res_list = []
    for length in slice_list:
        if length < 0:
            res_list.append([next(iter_list) for _ in range(flex_len)])
        else:
            res_list.append([next(iter_list) for _ in range(length)])
    if not single_item_as_list:
        res_list = [item[0] if is_list(item) and len(item) == 1 else item for item in res_list]
    return res_list


def set_seed(seed): # generated by AI
    random.seed(seed)                          # Python 内置随机模块
    np.random.seed(seed)                       # numpy
    torch.manual_seed(seed)                    # torch CPU
    torch.cuda.manual_seed(seed)               # torch GPU
    torch.cuda.manual_seed_all(seed)           # 多卡训练

    torch.backends.cudnn.deterministic = True  # 固定算法
    torch.backends.cudnn.benchmark = False     # 禁用加速优化，防止非确定性