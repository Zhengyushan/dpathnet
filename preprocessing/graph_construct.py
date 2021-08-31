#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2020/12
# author:Yushan Zheng
# emai:yszheng@buaa.edu.cn
import sys
sys.path.append('../thirdparty')


from torch.utils.data import DistributedSampler
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data.distributed
import torch.utils.data
import torch.multiprocessing as mp
import torch.optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import os
import argparse
import shutil
import time
import random
import warnings
import pickle
import cv2
import numpy as np
from yacs.config import CfgNode

from loader import SlideLocalTileDataset
from efficientnet_pytorch import EfficientNet
from utils import *

parser = argparse.ArgumentParser('Extract cnn freatures of whole slide images')
parser.add_argument('--cfg', type=str, default='',
                    help='The path of yaml config file')
parser.add_argument('--slide-dir', type=str,
                    default='/media/disk1/medical/gastric_slides')
parser.add_argument('--fold', type=int, default=0,
                    help='To identify the cnn used for feature extraction.\
                         a value -1 identify the cnn trained by all the training set data.\
                         It is useless when pretrained is set as True')

parser.add_argument('--arch', type=str, default='efficientnet-b0')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--resume', default='',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--dist-url', default='', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--fold-num', type=int, default=5)

parser.add_argument('--positive-ratio', type=float, default=0.5,
                    help='The image with positive pixels above the ratio will be labeled as positive sample')
parser.add_argument('--negative-ratio', type=float, default=0.05,
                    help='The image with negative pixels below the ratio will be labeled as negative sample')
parser.add_argument('--intensity-thred', type=int, default=25,
                    help='The threshold to recognize foreground regions')

parser.add_argument('--tile-size', type=int, default=512,
                    help='The size of local tile')
parser.add_argument('--level', type=int, default=1,
                    help='The layer index of the slide pyramid to sample')
parser.add_argument('--mask-level', type=int, default=3,
                    help='The layer index of the annotation mask')

parser.add_argument('--od-input', action='store_true', default=False)

parser.add_argument('--imsize', type=int, default=224,
                    help='The size of window.')
parser.add_argument('--step', type=int, default=112,
                    help='The step of window sliding for feature extraction.')
parser.add_argument('--batch-size', type=int, default=256)

parser.add_argument('--max-per-graph', type=int, default=512,
                    help='The upper bound of patches in a graph.')


def main(args):
    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    save_dir = get_feature_path(args)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.num_classes = len(binary_index) if args.label_id == 1 else 2

    args.gpu = gpu
    start_time = time.time()

    if args.gpu is not None:
        print("Use GPU: {} for encoding".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if 'efficientnet' in args.arch:
        if args.pretrained:
            model = EfficientNet.from_pretrained(
                args.arch, num_classes=args.num_classes)
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            model = EfficientNet.from_name(
                args.arch, num_classes=args.num_classes)
        # image_size = EfficientNet.get_image_size(args.arch)
    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch](num_classes=args.num_classes)

    if not args.pretrained:
        checkpoint = torch.load(os.path.join(get_cnn_path(
            args), 'model_best.pth.tar'), map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(
                    args.resume, map_location=torch.device('cpu'))
                model.load_state_dict(checkpoint['state_dict'])

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.num_workers = int(args.num_workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    print('Load model time', time.time() - start_time)

    if args.pretrained:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    model.eval()
    inference_module = model.module if args.distributed else model

    slide_list = get_slide_list_local(args.slide_dir)
    for s_id, s_guid in enumerate(slide_list):
        if args.distributed:
            # skip the slides the other gpus are working on
            if not s_id % ngpus_per_node == gpu:
                continue

        graph_save_path = os.path.join(get_graph_path(args), s_guid)
        if not os.path.exists(graph_save_path):
            os.makedirs(graph_save_path)

        graph_save_tag_file = os.path.join(
            get_graph_path(args), s_guid, 'info.txt')
        if os.path.exists(graph_save_tag_file):
            print('slide', s_guid, 'has completed. skip.')
            continue

        slide_path = os.path.join(args.slide_dir, s_guid)
        image_dir = os.path.join(slide_path, scales[args.level])

        mask = cv2.imread(os.path.join(slide_path, 'AnnotationMask.png'), 0)
        mask_mat = mask[(args.mhalfsize-1)::args.srstep,
                        (args.mhalfsize-1)::args.srstep]

        tissue_mask = get_tissue_mask(cv2.imread(
            os.path.join(slide_path, 'Overview.jpg')))
        content_mat = cv2.blur(
            tissue_mask, ksize=args.filter_size, anchor=(0, 0))
        content_mat = content_mat[::args.frstep,
                                  ::args.frstep] > args.intensity_thred

        roi_list_path = os.path.join(slide_path, 'BrowsingRecord.pkl')
        with open(roi_list_path, 'rb') as f:
            screen_list = pickle.load(f)
        path_len = len(screen_list)

        for roi_idx, sd in enumerate(screen_list):  # sd = (left, right, top, bottom)
            screen_rank_mat = np.zeros_like(mask)
            screen_rank_mat[sd[2]:sd[3], sd[0]:sd[1]] = 1
            screen_rank_mat = screen_rank_mat[(
                args.mhalfsize-1)::args.srstep, (args.mhalfsize-1)::args.srstep]

            patch_indexes = screen_rank_mat > 0
            patches_in_graph = np.sum(patch_indexes)
            if patches_in_graph < 1:
                continue

            # reduce the size of graph by grid sampling for the one that is larger than the upper bound
            down_factor = 1
            if patches_in_graph > args.max_per_graph:
                down_factor = int(
                    np.sqrt(patches_in_graph/args.max_per_graph)) + 1
                tmp = np.zeros(patch_indexes.shape, np.uint8) > 0
                tmp[::down_factor,
                    ::down_factor] = patch_indexes[::down_factor, ::down_factor]
                patch_indexes = tmp

            # patch position
            patch_pos = np.transpose(np.asarray(np.where(patch_indexes)))
            # ajdacency_mat
            adj = detect_connectivity(patch_pos, down_factor)
            # patch label
            patch_label = mask_mat[patch_indexes]
            # graph label
            tmp = np.bincount(patch_label)
            graph_label = np.argmax(tmp)

            # patch feature
            features = []
            if patch_pos.shape[0] > 0:
                image_dir = os.path.join(slide_path, scales[args.level])
                slide_dataset = SlideLocalTileDataset(image_dir, patch_pos*args.step, transform,
                                                      args.tile_size, args.imsize, od_mode=args.od_input)

                slide_loader = torch.utils.data.DataLoader(
                    slide_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, drop_last=False)

                # print(s_id, s_guid, 'roi', roi_idx, patch_pos.shape[0])

                for images in slide_loader:
                    images = images.cuda(args.gpu, non_blocking=True)
                    with torch.no_grad():
                        x = inference_module.extract_features(images)
                        ft = inference_module._avg_pooling(x)
                        ft = ft.flatten(start_dim=1)

                    features.append(ft.cpu().numpy())
                features = np.concatenate(features, axis=0)

                with open(os.path.join(graph_save_path, '{}.pkl'.format(roi_idx)), 'wb') as f:
                    graph = {'adj': adj,
                             'pos': patch_pos,
                             'down_factor': down_factor,
                             'feats': features,
                             'label': graph_label
                             }
                    pickle.dump(graph, f)

        with open(graph_save_tag_file, 'w') as f:
            f.write('path length: {}'.format(path_len))
        print(s_id, s_guid, 'number:', path_len)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
