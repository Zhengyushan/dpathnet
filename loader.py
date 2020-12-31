#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2020/12
# author:Yushan Zheng
# emai:yszheng@buaa.edu.cn

import numpy as np
import torch
import os
import pickle

# 0:LGIN, 1:HGIN, 2:Adenocarcinoma, 3:Mucinous adenocarcinoma, 4:SRCC
classes_task1 = np.asarray([0,1,2,3,4])
classes_task2 = np.asarray([0,0,1,1,1])

class DiagPathFileLoader(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''
    def __init__(self, graph_list_path, max_node_number, max_path_len, task_id=1, 
                    disable_adj=False, shuffle=False, reduce_rate=0):
        with open(graph_list_path, 'rb') as f:
            self.dl = pickle.load(f)
        
        with open(self.dl[0][0][0], 'rb') as f:
            graph_data = pickle.load(f)
        self.feat_dim = graph_data['feats'].shape[1]
        self.maxno = max_node_number
        self.ti = task_id
        self.type_num = 5 if task_id == 1 else 2
        self.use_adj = not disable_adj
        self.path_len = max_path_len
        self.shuffle = shuffle
        self.rr = reduce_rate

    def __len__(self):
        return len(self.dl)

    def __getitem__(self, idx):
        graph_paths = self.dl[idx][0]
        graph_num_nodes = np.zeros((self.path_len,))
        graph_feats = np.zeros((self.path_len,self.maxno,self.feat_dim))
        graph_adjs = np.zeros((self.path_len,self.maxno,self.maxno))
        
        if self.rr > 0:
            new_len = int(len(graph_paths)*self.rr)
            if new_len > 1:
                graph_paths = np.random.choice(graph_paths, new_len, replace=False)
        if self.shuffle:
            np.random.shuffle(graph_paths)

        for i, graph_name in enumerate(graph_paths):
            with open(graph_name, 'rb') as f:
                graph_data = pickle.load(f)
            
            num_node = min(graph_data['feats'].shape[0],self.maxno)
            graph_num_nodes[i] = num_node
            graph_adjs[i,:num_node, :num_node] = graph_data['adj'][:num_node,:num_node]\
                             if self.use_adj else np.zeros((num_node,num_node))
            graph_feats[i, :num_node, :] = graph_data['feats'][:num_node]

        graph_labels = np.asarray(self.dl[idx][1])-1
        if self.ti == 1:
            graph_labels = classes_task1[graph_labels]
        else:
            graph_labels = classes_task2[graph_labels]

        if self.ti==1:
            one_hot_label = np.max(np.eye(self.type_num)[graph_labels], axis=0)

            if np.sum(one_hot_label[2:])>0: # if the path contains cancer regions
                one_hot_label[:2]=0
            if one_hot_label[1]>0: # HGIN > LGIN
                one_hot_label[0]=0
        else:
            one_hot_label = np.eye(self.type_num)[np.max(graph_labels)]

        return graph_feats, graph_adjs, graph_num_nodes, np.sum(graph_num_nodes>0), one_hot_label

    def get_feat_dim(self):
        return self.feat_dim

    def get_max_node_number(self):
        return self.maxno

    def get_weights(self):
        num = self.__len__()
        labels = np.zeros((num,), np.int)
        for p_ind, path in enumerate(self.dl):
            labels[p_ind] = np.max(path[1])
        if self.ti == 1:
            labels = classes_task1[labels - 1]
        elif self.ti == 2:
            labels = classes_task2[labels - 1]

        tmp = np.bincount(labels)
        weights = 1 / np.asarray(tmp[labels], np.float)
        return weights

    def get_path_lengths(self):
        num = self.__len__()
        graph_number = np.zeros((num,), np.int)
        for p_ind, path in enumerate(self.dl):
            graph_number[p_ind] = len(path[1])

        return graph_number


class DistributedWeightedSampler(torch.utils.data.DistributedSampler):
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
