#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2020/12
# author:Yushan Zheng
# emai:yszheng@buaa.edu.cn

import os
import pickle
import cv2
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from utils import extract_tile


class PatchDataset(data.Dataset):
    def __init__(self, file_path, transform, od_mode=True, label_type=1):
        self.transform = transform
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        self.data_dir = data['base_dir']
        self.image_list = data['list']
        self.od = od_mode
        self.lt = label_type

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.image_list[index][0])).convert('RGB')
        label = self.image_list[index][self.lt]
        if self.transform!=None:
            img = self.transform(img)
        if self.od:
            img = -torch.log(img + 1.0/255.0)

        return img, label

    def __len__(self):
        return len(self.image_list)

    def get_weights(self):
        num = self.__len__()
        labels = np.zeros((num,), np.int)
        for s_ind, s in enumerate(self.image_list):
            labels[s_ind] = s[self.lt]
        tmp = np.bincount(labels)
        weights = 1.0 / np.asarray(tmp[labels], np.float)

        return weights


class SlideLocalTileDataset(data.Dataset):
    def __init__(self, image_dir, position_list, transform,
            tile_size=512, imsize=224, od_mode=False):
        self.transform = transform

        self.im_dir = image_dir
        self.pos = position_list
        self.od = od_mode
        self.ts = tile_size
        self.imsize = imsize

    def __getitem__(self, index):
        img = extract_tile(self.im_dir, self.ts, self.pos[index][1], self.pos[index][0], self.imsize, self.imsize)
        if len(img) == 0:
            img = np.ones((self.imsize, self.imsize, 3), np.uint8) * 240
        img = Image.fromarray(img[:,:,[2,1,0]]).convert('RGB')
        img = self.transform(img)

        if self.od:
            img = -torch.log(img + 1.0/255.0)

        return img

    def __len__(self):
        return self.pos.shape[0]


class DistributedWeightedSampler(data.DistributedSampler):
    def __init__(self, dataset, weights, num_replicas=None, rank=None, replacement=True):

        super(DistributedWeightedSampler, self).__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=False
            )
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, self.replacement).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
