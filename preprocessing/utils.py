#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2020/12
# author:yushan zheng
# emai:yszheng@buaa.edu.cn

import torch.nn.functional as F
import torch
import numpy as np
import os
import pickle
import cv2

# The definition of magnification of our gastic dataset.
# 'Large':40X, 'Medium':20X, 'Small':10X, 'Overview':5X
scales = ['Large', 'Medium', 'Small', 'Overview']

# The label in the gastric dataset:
# 0:normal, 1:LGIN, 2:HGIN, 3:Adenocarcinoma, 4:Mucinous adenocarcinoma, 5:SRCC
# types 3,4,5 are malignant tumors, which are considered as positive in the binary task. 
binary_index = [0,0,1,1,1,1]

def merge_config_to_args(args, cfg):
    # dirs
    args.patch_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'patch')
    args.list_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'patch_list')
    args.feat_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'cnn_feat')
    args.graph_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'graph')
    args.graph_list_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'graph_list')
    args.cnn_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'cnn_model')
    args.slide_dir = cfg.DATA.LOCAL_SLIDE_DIR

    # data
    args.label_id = cfg.DATA.LABEL_ID
    args.test_ratio = cfg.DATA.TEST_RATIO
    args.fold_num = cfg.DATA.FOLD_NUM

    # image
    if 'IMAGE' in cfg:
        args.level = cfg.IMAGE.LEVEL
        args.mask_level = cfg.IMAGE.MASK_LEVEL
        args.imsize = cfg.IMAGE.PATCH_SIZE
        args.tile_size = cfg.IMAGE.LOCAL_TILE_SIZE
        args.rl = args.mask_level-args.level
        args.msize = args.imsize >> args.rl
        args.mhalfsize = args.msize >> 1

    # sampling
    if 'SAMPLE' in cfg:
        args.positive_ratio = cfg.SAMPLE.POS_RAT
        args.negative_ratio = cfg.SAMPLE.NEG_RAT
        args.intensity_thred = cfg.SAMPLE.INTENSITY_THRED
        args.sample_step = cfg.SAMPLE.STEP
        args.max_per_class = cfg.SAMPLE.MAX_PER_CLASS
        args.save_mask = cfg.SAMPLE.SAVE_MASK
        
        args.srstep = args.sample_step>>args.rl
        args.filter_size = (args.imsize>>args.rl, args.imsize>>args.rl)
        
    # CNN
    if 'CNN' in cfg:
        args.arch = cfg.CNN.ARCH
        args.pretrained = cfg.CNN.PRETRAINED

    # feature
    if 'FEATURE' in cfg:
        args.step = cfg.FEATURE.STEP
        args.frstep = args.step>>args.rl
        
    # dignosis path
    if 'DPATH' in cfg:
        args.max_per_graph = cfg.DPATH.GRAPH_MAX_NODES
        args.max_per_path = cfg.DPATH.MAX_PER_PATH
    return args


def get_sampling_path(args):
    prefix = '[l{}t{}s{}m{}][p{}n{}i{}]'.format(args.level, args.imsize,
                                              args.step, args.max_per_class,
                                              int(args.positive_ratio * 100),
                                              int(args.negative_ratio * 100),
                                              args.intensity_thred)

    return os.path.join(args.patch_dir, prefix)

def get_data_list_path(args):
    prefix = get_sampling_path(args)
    prefix = '{}[f{}_t{}]'.format(prefix[prefix.find('['):], args.fold_num,
                                int(args.test_ratio * 100))

    return os.path.join(args.list_dir, prefix)


def get_cnn_path(args):
    prefix = get_data_list_path(args)
    args.fold_name = 'list_fold_all' if args.fold == -1 else 'list_fold_{}'.format(
        args.fold)
    prefix = '{}[{}_td_{}_{}]'.format(prefix[prefix.find('['):], args.arch, args.label_id,
                                    args.fold_name)


    return os.path.join(args.cnn_dir, prefix)


def get_feature_path(args):
    if args.pretrained:
        prefix = '[{}_pre][fs{}]'.format(args.arch, args.step)
    else:
        prefix = get_data_list_path(args)
        args.fold_name = 'list_fold_all' if args.fold == -1 else 'list_fold_{}'.format(
                        args.fold)
        prefix = '{}[{}_td_{}][fs{}][{}]'.format(prefix[prefix.find('['):], 
            args.arch, args.label_id, args.step, args.fold_name)

    return os.path.join(args.feat_dir, prefix)


def get_graph_path(args):
    if args.pretrained:
        prefix = '[{}_pre][fs{}_m{}]'.format(args.arch, args.step, args.max_per_graph)
    else:
        prefix = get_data_list_path(args)
        args.fold_name = 'list_fold_all' if args.fold == -1 else 'list_fold_{}'.format(
                        args.fold)
        prefix = '{}[{}_td_{}][fs{}_m{}][{}]'.format(prefix[prefix.find('['):], 
            args.arch, args.label_id, args.step, args.max_per_graph, args.fold_name)

    return os.path.join(args.graph_dir, prefix)


def get_graph_list_path(args):
    if args.pretrained:
        prefix = '[{}_pre][fs{}_m{}][pl{}]'.format(args.arch, args.step,
                args.max_per_graph, args.max_per_path)
    else:
        prefix = get_data_list_path(args)
        prefix = '{}[{}_td_{}][fs{}_m{}][pl{}]'.format(prefix[prefix.find('['):], 
            args.arch, args.label_id, args.step, args.max_per_graph, args.max_per_path)

    return os.path.join(args.graph_list_dir,prefix)


def get_slide_list_local(slide_dir):
    slides = os.listdir(slide_dir)
    slide_list = []
    for s_id, s_guid in enumerate(slides):
        # the slides in our dataset are named by guids
        if len(s_guid) < 36:
            continue

        slide_path = os.path.join(slide_dir, s_guid)
        slide_content = os.listdir(slide_path)
        # Check data integrity
        if len(slide_content) < 11:
            print(s_id, s_guid, 'is incomlete. skip.')
            continue

        slide_list.append(s_guid)

    return slide_list


def extract_tile(image_dir, tile_size, x, y, width, height):
    x_start_tile = x // tile_size
    y_start_tile = y // tile_size
    x_end_tile = (x+width) // tile_size
    y_end_tile = (y+height) // tile_size

    tmp_image = np.zeros(
        ((y_end_tile-y_start_tile+1)*tile_size, (x_end_tile-x_start_tile+1)*tile_size, 3),
        np.uint8)

    for y_id, col in enumerate(range(x_start_tile, x_end_tile + 1)):
        for x_id, row in enumerate(range(y_start_tile, y_end_tile + 1)):
            img_path = os.path.join(image_dir, '{:04d}_{:04d}.jpg'.format(row,col))
            if not os.path.exists(img_path):
                return []
            tmp_image[(x_id*tile_size):(x_id+1)*tile_size, (y_id*tile_size):(y_id+1)*tile_size,:] = \
                cv2.imread(img_path)

    x_off = x % tile_size
    y_off = y % tile_size
    output = tmp_image[y_off:y_off+height, x_off:x_off+width]
    return output


def get_tissue_mask(wsi_thumbnail, scale=30):
    hsv = cv2.cvtColor(wsi_thumbnail, cv2.COLOR_RGB2HSV)
    _, tissue_mask = cv2.threshold(hsv[:, :, 2], 210, 255, cv2.THRESH_BINARY_INV)
    tissue_mask[hsv[:, :, 0]<10]=0

    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * scale + 1, 2 * scale + 1)
        )
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, element)

    return tissue_mask

def detect_connectivity(positions, down_factor):
    power = np.sum(np.multiply(positions, positions), axis=1)
    power = np.repeat(power[np.newaxis, :], positions.shape[0], axis=0)
    dist_map = np.abs(power - 2*np.dot(positions, np.transpose(positions)) + np.transpose(power))
    adj_mat = dist_map <= down_factor*down_factor

    return adj_mat

def accuracy(output, target, topk=(1,2)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
