#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/12/14
"""
import argparse

import torch

from config.default_config import get_default_config, imagedata_kwargs, optimizer_kwargs, lr_scheduler_kwargs, \
    engine_run_kwargs
from dataset.datamanager import ImageDataManager
from engine.softmax import ImageSoftmaxEngine
from engine.triplet import ImageTripletEngine
from model.osnet import osnet_x1_0
from utils.tools import set_random_seed, load_pretrained_weights, build_optimizer, build_scheduler


def build_data_manager(cfg):
    return ImageDataManager(**imagedata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler):
    if cfg.loss.name == 'softmax':
        engine = ImageSoftmaxEngine(
            datamanager,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            use_gpu=cfg.use_gpu,
            label_smooth=cfg.loss.softmax.label_smooth
        )

    else:
        engine = ImageTripletEngine(
            datamanager,
            model,
            optimizer=optimizer,
            margin=cfg.loss.triplet.margin,
            weight_t=cfg.loss.triplet.weight_t,
            weight_x=cfg.loss.triplet.weight_x,
            scheduler=scheduler,
            use_gpu=cfg.use_gpu,
            label_smooth=cfg.loss.softmax.label_smooth
        )
    return engine


def train(cfg):

    set_random_seed(cfg.train.seed)
    torch.backends.cudnn.benchmark = True
    # data_manager
    data_manager = build_data_manager(cfg)

    model = osnet_x1_0(num_classes=data_manager.num_train_pids, loss=cfg.loss.name, pretrained=cfg.model.pretrained,
                       use_gpu=cfg.use_gpu)
    if cfg.model.load_weights:
        load_pretrained_weights(model, cfg.model.load_weights)
    model = model.cuda()
    optimizer = build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = build_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    # print(model)
    engine = build_engine(cfg, data_manager, model, optimizer, scheduler)
    engine.run(**engine_run_kwargs(cfg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, default='./config/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml')
    args = parser.parse_args()
    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    train(cfg)
