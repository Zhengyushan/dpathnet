#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2020/12
# author:Yushan Zheng
# emai:yszheng@buaa.edu.cn

import numpy as np
import os
import argparse
import pickle
from yacs.config import CfgNode
from utils import *

parser = argparse.ArgumentParser('Make diagnosis path data list')
parser.add_argument('--cfg', type=str, default='', help='The path of yaml config file')
parser.add_argument('--fold', type=int, default=0,
                    help='To identify the cnn used for feature extraction.\
                         a value -1 identify the cnn trained by all the training set data.\
                         It is useless when pretrained is set as True')

args = parser.parse_args()

def main():
    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

    diagnosis_path_list_path = get_graph_list_path(args)
    if not os.path.exists(diagnosis_path_list_path):
        os.makedirs(diagnosis_path_list_path)

    if not args.pretrained:
        with open(os.path.join(get_data_list_path(args), 'split.pkl'), 'rb') as f:
            folds = pickle.load(f)
    else:
        dataset_split_path = os.path.join(diagnosis_path_list_path, 'split.pkl')
        if not os.path.exists(dataset_split_path):
            slide_list = get_slide_list_local(args.slide_dir)
            np.random.shuffle(slide_list)
            test_list = slide_list[:int(len(slide_list)*args.test_ratio)]
            train_list = slide_list[int(len(slide_list)*args.test_ratio):]

            folds = []
            for f_id in range(args.fold_num):
                folds.append(train_list[f_id::args.fold_num])
            folds.append(test_list)

            with open(dataset_split_path, 'wb') as f:
                pickle.dump(folds, f)
        else:
            with open(dataset_split_path, 'rb') as f:
                folds = pickle.load(f)

    graph_dir = get_graph_path(args)

    args.num_classes = len(binary_index)
    sample_list = []
    for f_id, f_list in enumerate(folds):
        class_graph_counter = np.zeros(args.num_classes)
        sample_list_fold = []
        for s_guid in f_list:
            slide_dir = os.path.join(graph_dir, s_guid)
            if not os.path.isdir(slide_dir):
                continue

            graph_list = os.listdir(slide_dir)
            roi_rank = []
            for g_name in graph_list:
                if g_name[-4:] == '.pkl':
                    roi_rank.append(int(g_name[:-4]))
            if not len(roi_rank):
                continue

            path_graph_list_fold = []
            graph_labels = []
            for g_name in range(min(roi_rank),max(roi_rank)+1):
                graph_path = os.path.join(slide_dir, str(g_name) + '.pkl')
                if not os.path.exists(graph_path):
                    continue
                with open(graph_path, 'rb') as f:
                    graph = pickle.load(f)

                c_index = graph['label']

                path_graph_list_fold.append(graph_path)
                graph_labels.append(c_index)
                class_graph_counter[c_index] += 1
                
            for i in range(0, len(graph_labels), args.max_per_path):
                tmp_len = args.max_per_path
                if np.sum(graph_labels[i:i+tmp_len]) > 0:
                    sample_list_fold.append(
                        (path_graph_list_fold[i:i+tmp_len], graph_labels[i:i+tmp_len])
                    )
            print(s_guid, len(graph_labels))

        with open(os.path.join(diagnosis_path_list_path, 'list_config.txt'), 'a') as f:
            print_str = ' graph number: '
            for num in class_graph_counter:
                print_str += '{},'.format(num)
            print_str += '\n'
            f.write(print_str)
        
        sample_list.append(sample_list_fold)


    for f_id in range(args.fold_num+1):
        f_name = 'list_fold_all' if f_id==args.fold_num else 'list_fold_{}'.format(f_id)
        if not args.pretrained:
            if (f_name != 'list_fold_all') and (f_id != args.fold):  
                print(args.fold, f_name)
                continue
            
        val_set = sample_list[f_id]

        train_set = []
        if f_id == args.fold_num:
            for train_f_id in range(args.fold_num+1):
                train_set += sample_list[train_f_id]
        else:
            train_index = np.hstack((np.arange(0,f_id),np.arange(f_id+1,args.fold_num)))
            for train_f_id in train_index:
                train_set += sample_list[train_f_id]

        test_set = sample_list[-1]
        list_dir = os.path.join(diagnosis_path_list_path, f_name)
        if not os.path.exists(list_dir):
            os.makedirs(list_dir)

        with open(os.path.join(list_dir,'train'), 'wb') as f:
            pickle.dump(train_set, f)
        if len(val_set):
            with open(os.path.join(list_dir,'val'), 'wb') as f:
                pickle.dump(val_set, f)
        if len(test_set):
            with open(os.path.join(list_dir,'test'), 'wb') as f:
                pickle.dump(test_set, f)


    return 0
    
if __name__ == "__main__":
    main()
