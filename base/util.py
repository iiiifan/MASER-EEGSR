'''
This is written by Jiyuan Liu, Dec. 21, 2021.
Homepage: https://liujiyuan13.github.io.
Email: liujiyuan13@163.com.
All rights reserved.
'''

import os
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn import metrics

def set_seed(seed=0):
    """
    set seed for torch.
    @param seed: int, default 0
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def save_ckpt(model, optimizer, args, epoch, save_file):
    '''
    save checkpoint
    :param model: target model
    :param optimizer: used optimizer
    :param args: training parameters
    :param epoch: save at which epoch
    :param save_file: file path
    :return:
    '''
    ckpt = {
        # 'args': args,
        'model': model,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(ckpt, save_file)
    del ckpt


def load_ckpt(model, load_file):
    '''
    load ckpt to model
    :param model: target model
    :param load_file: file path
    :return: the loaded model
    '''
    ckpt = torch.load(load_file)
    model.load_state_dict(ckpt['model'])
    del ckpt
    return model

class AverageMeter(object):
    '''
    compute and store the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count