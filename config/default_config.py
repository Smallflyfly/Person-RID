#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/12/14
"""
from yacs.config import CfgNode


def get_default_config():
    cfg = CfgNode()

    # model
    cfg.model = CfgNode()
    cfg.model.name = 'osnet'
    # pretrained weights
    cfg.model.pretrained = ''
    cfg.model.resume = ''

    # data
    cfg.data.type = 'image'
    cfg.data.root = 'data'
