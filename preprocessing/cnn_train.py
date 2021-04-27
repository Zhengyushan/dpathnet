#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Modified from the training code for EfficientNet by 
Author: lukemelas (github username)
Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
"""
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
from yacs.config import CfgNode
from sklearn import metrics
import numpy as np
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import pickle

from utils import *
from loader import PatchDataset, DistributedWeightedSampler
from efficientnet_pytorch import EfficientNet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--cfg', type=str, default='',
                    help='The path of yaml config file')
parser.add_argument('--data', type=str, default='',)
parser.add_argument('--fold', type=int, default=0,
                    help='use all data for training if it is set -1')

parser.add_argument('--num-classes', default=6, type=int, metavar='N',
                    help='class number of in the dataset')
parser.add_argument('--val-ratio', default=0.0, type=float,
                    help='ratio of train data for validation')
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b0',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=False,
                    help='use pre-trained model')
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
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image-size', default=224, type=int,
                    help='image size')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--label-id', type=int, default=1,
                    help='1 for all type classification, 2 for binary classification')

parser.add_argument('--od-input', action='store_true', default=False)
parser.add_argument('--weighted-sample', action='store_true')

parser.add_argument('--redo', action='store_true', default=False,
                    help='Ignore all the existing results and caches.')

best_acc1 = 0


def main(args):
    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

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
    args.num_classes = len(binary_index) if args.label_id == 1 else 2

    checkpoint = []
    model_save_dir = get_cnn_path(args)
    if not args.redo:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            print("=> loading checkpoint '{}'".format(args.resume))
        else:
            checkpoint_path = os.path.join(
                model_save_dir, 'checkpoint.pth.tar')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(
                    checkpoint_path, map_location=torch.device('cpu'))
                print("=> loading checkpoint")

    if checkpoint:
        args.start_epoch = checkpoint['epoch']
        if args.start_epoch >= args.epochs:
            print('CNN training is finished')
            return 0
        else:
            print('CNN train from epoch {}/{}'.format(args.start_epoch, args.epochs))

    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

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
    # create model
    if 'efficientnet' in args.arch:  # NEW
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
            args.workers = int(args.workers / ngpus_per_node)
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

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # optionally resume from a checkpoint
    if checkpoint:
        best_acc1 = checkpoint['best_acc1']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    list_dir = args.data if len(args.data) else os.path.join(
        get_data_list_path(args), args.fold_name)
    traindir = os.path.join(list_dir, 'train')
    valdir = os.path.join(list_dir, 'val')
    testdir = os.path.join(list_dir, 'test')

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.1, 0.3, 0.1, 0.1),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Datasets
    train_set = PatchDataset(traindir,
                             od_mode=args.od_input,
                             transform=train_transforms,
                             label_type=args.label_id
                             )
    if os.path.exists(valdir):
        valid_set = PatchDataset(valdir,
                                 od_mode=args.od_input,
                                 transform=test_transforms,
                                 label_type=args.label_id
                                 )
    elif args.val_ratio:
        valid_set = PatchDataset(traindir,
                                 od_mode=args.od_input,
                                 transform=test_transforms,
                                 label_type=args.label_id
                                 )
        indices_file_path = os.path.join(model_save_dir, 'train_indices')
        if os.path.exists(indices_file_path):
            with open(indices_file_path, 'rb') as f:
                indices = pickle.load(f)
        else:
            indices = torch.randperm(len(train_set))
            with open(indices_file_path, 'wb') as f:
                pickle.dump(indices, f)

        train_indices = indices[int(len(indices) * args.val_ratio):]
        valid_indices = indices[:int(len(indices) * args.val_ratio)]
        train_set = torch.utils.data.Subset(train_set, train_indices)
        valid_set = torch.utils.data.Subset(valid_set, valid_indices)
    else:
        valid_set = None

    if os.path.exists(testdir):
        test_set = PatchDataset(testdir,
                                od_mode=args.od_input,
                                transform=test_transforms,
                                label_type=args.label_id
                                )
    else:
        test_set = None

    if args.distributed:
        if args.weighted_sample:
            print('activate weighted sampling')
            train_sampler = DistributedWeightedSampler(
                train_set, train_set.get_weights())
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_set)

        if test_set is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_set)

        if valid_set is not None:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_set)
    else:
        if args.weighted_sample:
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                train_set.get_weights(), len(train_set), replacement=True
            )
        else:
            train_sampler = None
        val_sampler = None
        test_sampler = None

    if test_set is not None:
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    if args.evaluate:
        res = validate(test_loader, model, criterion, args)
        with open('res.txt', 'w') as f:
            print(res, file=f)
        return

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler
    )
    if valid_set is not None:
        val_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler
        )

    with open(model_save_dir + '.csv', 'w') as f:
        f.write('epoch, train acc,val acc, test acc\n')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_acc = train(train_loader, model, criterion,
                          optimizer, epoch, args)

        # evaluate on validation set
        if valid_set is not None:
            acc1, val_cm = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
        else:
            is_best = train_acc > best_acc1
            best_acc1 = max(train_acc, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_data = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(save_data, False,
                            os.path.join(model_save_dir, 'checkpoint.pth.tar'))

            if is_best:
                if test_set is not None:
                    test_acc, test_cm = validate(
                        test_loader, model, criterion, args)
                else:
                    test_acc = acc1
                    test_cm = val_cm

                with open(model_save_dir + '.csv', 'a') as f:
                    f.write('{},{:.2f},{:.2f},{:.2f}'.format(
                        epoch, train_acc, best_acc1, test_acc))
                    for cn in range(test_cm.shape[0]):
                        f.write(',{:.2f}'.format(test_cm[cn, cn]))
                    f.write('\n')

                save_single_module = {
                    'arch': args.arch,
                    'state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                    'pretrained': args.pretrained,
                    'level': args.level,
                    'input_size': args.imsize,
                    'intensity_thred': args.intensity_thred
                }
                save_checkpoint(
                    save_single_module, is_best,
                    os.path.join(model_save_dir, 'model_single_{}.pth.tar'.format(epoch)))

                with open(os.path.join(model_save_dir, 'result_{}'.format(epoch)), 'wb') as f:
                    pickle.dump({'cm': test_cm, 'acc': test_acc}, f)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top2, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc2 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top2.update(acc2[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    return top1.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top2,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    y_preds = []
    y_labels = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            y_preds.append(output.cpu().data)
            y_labels.append(target.cpu().data)
            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top2.update(acc2[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top2.avg:.3f}'
              .format(top1=top1, top2=top2))

    y_preds = np.concatenate(y_preds, axis=0)
    y_labels = np.hstack(y_labels)

    confuse_mat = metrics.confusion_matrix(
        y_labels, np.argmax(y_preds, axis=1))
    confuse_mat = np.asarray(confuse_mat, np.float)
    for y in range(max(y_labels)+1):
        confuse_mat[y, :] = confuse_mat[y, :]/np.sum(y_labels == y)

    return top1.avg, confuse_mat


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(
            os.path.dirname(filename), 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    decay_factor = 2 if epoch > args.epochs//2 else 1
    decay_factor = decay_factor//2 if epoch > int(
        args.epochs*0.75) else decay_factor

    lr = args.lr * (0.1 ** decay_factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
