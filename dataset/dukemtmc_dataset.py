#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/12/14
"""
import os
import re
from glob import glob

from dataset.dataset import ImageDataset


class DukeMTMCreID(ImageDataset):
    """DukeMTMC-reID.

       Reference:
           - Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
           - Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.

       URL: `<https://github.com/layumi/DukeMTMC-reID_evaluation>`_

       Dataset statistics:
           - identities: 1404 (train + query).
           - images:16522 (train) + 2228 (query) + 17661 (gallery).
           - cameras: 8.
       """
    dataset_dir = 'data'

    def __init__(self, train, query, gallery, root='', **kwargs):
        super().__init__(train, query, gallery, **kwargs)
        self.root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.train_dir = os.path.join(os.path.join(self.dataset_dir, 'DukeMTMC-reID'), 'bounding_box_train')
        self.query_dir = os.path.join(os.path.join(self.query_dir, 'DukeMTMC-reID'), 'query')
        self.gallery_dir = os.path.join(os.path.join(self.dataset_dir, 'DukeMTMC-reID'), 'bounding_box_test')
        required_files = [self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir]
        self.check_before_run(required_files)
        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dirm, relabel=False)


    def process_dir(self, dir_path, relabel=False):
        img_path = glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')