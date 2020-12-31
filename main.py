#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2020/12
# author:Yushan Zheng
# emai:yszheng@buaa.edu.cn

import argparse
import os
import pickle
import random
import warnings
import shutil
import time
import sys
import builtins
import numpy as np
from tabulate import tabulate

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import DRANet
from loader import DiagPathFileLoader, DistributedWeightedSampler
from retrieval_utils import *


def arg_parse():
    parser = argparse.ArgumentParser(description='DRA-Net arguments.')

    # basic settings
    parser.add_argument('--dataset-dir', type=str,
                        help='Directory of the graph list')
    parser.add_argument('--result-dir', type=str, default='./data',
                        help='Directory to save the model and results')
    parser.add_argument('--prefix-name', type=str, default='DRA-Net')

    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')

    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--num-epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='The epoch to start.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers to load data.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='Weight decay.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum.')
    parser.add_argument('--clip', type=float, default=2.0,
                        help='Gradient clipping.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--shuffle-train', default=False, action='store_true')

    parser.add_argument('--task-id', type=int, default=1,
                        help='1: five categoriy task, 2: binary task')
    parser.add_argument('--hash-bits', type=int, default=32)

    parser.add_argument('--redo', default=False, action='store_true')
    parser.add_argument('--eval-model', type=str, default='',
                        help='provide a path of a trained model to evaluate the performance')
    parser.add_argument('--eval-freq', type=int, default=10)

    # Diffpool GCN configuration
    parser.add_argument('--assign-ratio', type=float, default=0.2,
                        help='ratio of number of nodes in consecutive layers')
    parser.add_argument('--num-pool', type=int, default=2,
                        help='number of pooling layers')
    parser.add_argument('--linkpred', action='store_true', default=False,
                        help='Whether link prediction side objective is used')
    parser.add_argument('--max-nodes', type=int, default=300,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--num-gc-layers', type=int, default=3,
                        help='Number of hidden layers in each GCN')
    parser.add_argument('--hidden-dim', type=int, default=110,
                        help='Dimension of GCN hidden layers')
    parser.add_argument('--output-dim', type=int, default=50,
                        help='Dimension of GCN output layer')
    parser.add_argument('--nobn', dest='bn', action='store_const', const=False,
                        default=True, help='Whether batch normalization is used')
    parser.add_argument('--nobias', dest='bias', action='store_const', const=False, default=True,
                        help='Whether to add bias. Default to True.')

    # RNN configuration
    parser.add_argument('--max-sq-len', type=int, default=64,
                        help='The maximal length of the path considered in the training.')
    parser.add_argument('--rnn-model', type=str, default='LSTM',
                        help='Use RNN to model the temporal information of graphs, support RNN, LSTM, GRU')
    parser.add_argument('--num-rnn-layers', type=int, default=1)
    parser.add_argument('--num-rnn-hdim', type=int, default=128)

    # Ablation configuration. Please refer to the paper for details.
    parser.add_argument('--disable-att', default=False, action='store_true',
                        help='Disable the attention module')
    parser.add_argument('--disable-adj', default=False, action='store_true',
                        help='Disable the adjacency matrix in the GCN.')
    parser.add_argument('--shuffle-seq', default=False, action='store_true',
                        help='Randomly change the order of ROIs within each path.')
    parser.add_argument('--reduce-rate', type=float, default=0.0,
                        help='Randomly reduce the number of ROIs within each path.')

    return parser.parse_args()


def main():
    args = arg_parse()
    args.num_classes = 5 if args.task_id == 1 else 2
    args.prefix = gen_prefix(args)
    args.model_dir = os.path.join(args.result_dir, 'dra_model', args.prefix)
    if not os.path.exists(args.model_dir):
        print('Creating', args.model_dir)
        os.makedirs(args.model_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
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
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    checkpoint = None
    if not args.redo:
        checkpoint_path = os.path.join(args.model_dir, 'checkpoint.pth.tar')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print("=>DRA-Net loading checkpoint")

            args.start_epoch = checkpoint['epoch']
            if args.start_epoch >= args.num_epochs:
                print('DRA-Net training is finished')
                return 0
            else:
                print(
                    'DRA-Net train from epoch {}/{}'.format(args.start_epoch, args.num_epochs))

    if args.gpu is not None and not args.distributed:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # train graph data
    train_set = DiagPathFileLoader(
        os.path.join(args.dataset_dir, 'train'),
        max_node_number=args.max_nodes,
        max_path_len=args.max_sq_len,
        task_id=args.task_id,
        disable_adj=args.disable_adj,
        shuffle=args.shuffle_seq,
        reduce_rate=args.reduce_rate
    )
    args.input_dim = train_set.get_feat_dim()

    # create model
    model = DRANet(args.num_classes, args.hash_bits,
                   args.rnn_model, args.num_rnn_layers, args.num_rnn_hdim,
                   args.max_sq_len, args.dropout, args.disable_att, args
                   )

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.num_workers = int(args.num_workers / ngpus_per_node)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True)
    elif args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    if args.distributed:
        train_sampler = DistributedWeightedSampler(train_set,
                                                   train_set.get_weights(), replacement=True
                                                   )
    else:
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            train_set.get_weights(), len(train_set), replacement=True
        )

    cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=args.shuffle_train,
        num_workers=args.num_workers, sampler=train_sampler)

    # the loader for retrieval database
    database_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # validation graph data
    val_path = os.path.join(args.dataset_dir, 'val')
    if not os.path.exists(val_path):
        valid_loader = None
    else:
        valid_set = DiagPathFileLoader(val_path,
                                       max_node_number=args.max_nodes,
                                       max_path_len=args.max_sq_len,
                                       task_id=args.task_id,
                                       disable_adj=args.disable_adj,
                                       shuffle=args.shuffle_seq,
                                       reduce_rate=args.reduce_rate
                                       )
        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
        )

    # test graph data
    test_path = os.path.join(args.dataset_dir, 'test')
    if not os.path.exists(test_path):
        test_loader = None
    else:
        test_set = DiagPathFileLoader(test_path,
                                      max_node_number=args.max_nodes,
                                      max_path_len=args.max_sq_len,
                                      task_id=args.task_id,
                                      disable_adj=args.disable_adj,
                                      shuffle=args.shuffle_seq,
                                      reduce_rate=args.reduce_rate
                                      )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)

    with open(args.model_dir + 'results.csv', 'w') as f:
        f.write('epoch,V,val_p,val_map,val_mrr,T,test_p5,test_p20,test_mrr,test_map\n')

    if len(args.eval_model) > 0:
        model_params = torch.load(args.eval_model)
        model.load_state_dict(model_params['state_dict'])
        db_label, db_code = hash_encoding(database_loader, model)
        if valid_loader is not None:
            val_label, val_code = hash_encoding(valid_loader, model)
            vmet = evaluate(val_code, val_label, db_code,
                            db_label, 'Val')

        if test_loader is not None:
            test_label, test_code = hash_encoding(valid_loader, model)
            tmet = evaluate(test_code, test_label, db_code,
                            db_labelodel, 'Test')

        if valid_loader is not None and test_loader is not None:
            with open(args.model_dir + 'results.csv', 'a') as f:
                f.write('Evaluate,V,%0.3f,%0.3f,%0.3f,%0.3f,T,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f\n' % (
                    vmet['ap5'], vmet['mrr'], vmet['map'], vmet['r5'],
                    tmet['ap5'], tmet['ap20'], tmet['mrr'], tmet['map'], tmet['r5'])
                )
        return 0

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                nesterov=True, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[0.5 * args.num_epochs, 0.75 * args.num_epochs], gamma=0.1
    )

    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    for epoch in range(args.start_epoch, args.num_epochs):
        begin_time = time.time()
        train_loss = train(train_loader, model, optimizer, args)

        scheduler.step()
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            print('Epoch: {} Avg loss: {:.3f} epoch time: {:.3f}'.format(
                epoch, train_loss, time.time() - begin_time))

            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                        },
                       os.path.join(args.model_dir, 'checkpoint.pth.tar'))

            if epoch % args.eval_freq == 0:
                db_label, db_code = hash_encoding(database_loader, model)
                if valid_loader is not None:
                    val_label, val_code = hash_encoding(valid_loader, model)
                    vmet = evaluate(val_code, val_label,
                                    db_code, db_label, 'Val')

                if test_loader is not None:
                    test_label, test_code = hash_encoding(valid_loader, model)
                    tmet = evaluate(test_code, test_label,
                                    db_code, db_label, 'Test')

                if valid_loader is not None and test_loader is not None:
                    with open(args.model_dir + 'results.csv', 'a') as f:
                        f.write('%03d,V,%0.3f,%0.3f,%0.3f,T,%0.3f,%0.3f,%0.3f,%0.3f\n' % (
                            epoch + 1, vmet['ap5'], vmet['mrr'], vmet['map'],
                            tmet['ap5'], tmet['ap20'], tmet['mrr'], tmet['map'])
                        )
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
                           os.path.join(args.model_dir, 'graph_model_ep{}.dat'.format(epoch+1)))


def train(train_loader, model, optimizer, args):
    avg_loss = 0.0
    model.train()
    for data in train_loader:
        cls_predict, hash_code, _ = model(data)
        label = data[4].view(-1, data[4].size()[-1]).cuda()
        '''
            Here, we append a classification loss function to accelerate
            the training of the network and meanwhile enable it to classify
            the wsi.
        '''
        loss = triplet_loss(hash_code, label,
                            margin=args.hash_bits >> 1
                            ) + 100 * F.mse_loss(label, torch.sigmoid(cls_predict))

        # backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        avg_loss += loss.cpu().data.numpy()
    avg_loss /= len(train_loader)

    return avg_loss


### utils
def evaluate(query_code, query_label, db_code, db_label, prefix='Eval'):
    _, correct = retrieval(query_code.clone(), query_label,
                           db_code.clone(), db_label)
    correct = correct.numpy()
    ap_5 = average_precision(correct, ret_num=5)
    ap_20 = average_precision(correct, ret_num=20)
    mAP = mean_average_precision(correct)[-1]
    mRR = mean_reciprocal_rank(correct)
    r5 = recall_at_n(correct, ret_num=5)

    print(tabulate([[prefix, ap_5, ap_20, mRR, mAP, r5]],
                   headers=['CBIR', 'p@5', 'p@20', 'MRR', 'MAP', 'R5'], tablefmt="grid")
          )

    return {'ap5': ap_5, 'ap20': ap_20, 'map': mAP, 'mrr': mRR, 'r5': r5}


def hash_encoding(dataset, model):
    labels = []
    codes = []

    model.eval()
    with torch.no_grad():
        for data in dataset:
            _, sq_hash_code, _ = model(data)

            labels.append(data[4].view(-1, data[4].size()[-1]))
            codes.append(sq_hash_code.data.cpu())

    labels = torch.cat(labels, axis=0)
    codes = torch.cat(codes, axis=0)

    return labels, codes


def triplet_loss(features, label, margin, hash_weights=None):
    # Implementation of the triplet hashing loss referring to
    # "Wang et al. Deep supervised hashing with triplet labels, ACCV 2016"

    hash_bits = features.size()[1]
    batch_size = features.size()[0]

    cross_sim_mat = torch.matmul(features, concat_all_gather(features).T) / 2
    cross_label_mat = torch.matmul(label, concat_all_gather(label).T)

    # search the hardest negative for each sample
    tmp = cross_sim_mat.clone().detach()
    tmp[cross_label_mat > 0] = -hash_bits
    idx = torch.argmax(tmp, dim=1)
    an_product = cross_sim_mat.index_select(1, idx).diagonal()

    # search the hardest postive for each sample
    tmp = cross_sim_mat.clone().detach()
    tmp[cross_label_mat < 1] = hash_bits
    idx = torch.argmin(tmp, dim=1)
    ap_product = cross_sim_mat.index_select(1, idx).diagonal()

    loss = - torch.sum(ap_product-an_product-margin
                       - torch.log(1 + torch.exp(ap_product-an_product-margin))
                       ) \
        + 0.01 * torch.sum(torch.pow(features-torch.sign(features), 2.0))

    loss /= batch_size

    return loss


@torch.no_grad()
def concat_all_gather(tensor):
    if torch.distributed.is_initialized():
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
    else: 
        output = tensor

    return output


def gen_prefix(args):
    name = '{}[{}][l_{}x{}_ar_{:.2f}_hs_{}_os_{}][{}_l_{}_hs_{}][hb_{}][cls_{}][att_{}_adj_{}_sff_{}_rr_{:.2f}]'.format(
        args.prefix_name,
        args.dataset_dir[-6:],
        args.num_gc_layers,
        args.num_pool,
        args.assign_ratio,
        args.hidden_dim,
        args.output_dim,
        args.rnn_model,
        args.num_rnn_layers,
        args.num_rnn_hdim,
        args.hash_bits,
        args.num_classes,
        'n' if args.disable_att else 'y',
        'n' if args.disable_adj else 'y',
        'y' if args.shuffle_seq else 'n',
        args.reduce_rate
    )

    return name


if __name__ == "__main__":
    main()
