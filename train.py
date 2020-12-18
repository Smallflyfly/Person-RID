#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/12/14
"""
import torch

from config.default_config import get_default_config
from utils.tools import set_random_seed


def train():
    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    set_random_seed(cfg.train.seed)
    model = build_model(

    )


if __name__ == '__main__':
    train()