#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2020/12
# author:Yushan Zheng
# emai:yszheng@buaa.edu.cn

import torch.nn.functional as F
import torch
import numpy as np
import os

# Evaluation function
def retrieval(query, query_label, database, database_label):
    query[query > 0] = 1
    query[query < 0] = -1
    database[database > 0] = 1
    database[database < 0] = -1

    if len(query.size()) > 2:
        hamming_distance = torch.einsum('nsk,mqk->nmsq', query, database) / 2
        hamming_distance = torch.mean(hamming_distance, dim=(2,3))
    else:
        hamming_distance = torch.matmul(query, database.T)
    sim_mat = torch.matmul(query_label, database_label.T).int()

    ret_index = torch.argsort(hamming_distance, axis=1, descending=True)
    _, inv_index = ret_index.sort()
    correct = sim_mat.clone().scatter_(
        1, inv_index, sim_mat) > 0

    return ret_index, correct.int()


# Retrieval Metrics
def mean_average_precision(correct_mat):
    tmp_mat = np.asarray(correct_mat, np.int32)

    ave_p = np.cumsum(tmp_mat, axis=1) / np.arange(1,tmp_mat.shape[1]+1)
    ave_p_tmp = ave_p.copy()
    ave_p_tmp[tmp_mat < 1] = 0

    mean_ave_p = np.cumsum(ave_p_tmp, axis=1) / (np.cumsum(tmp_mat, axis=1) + 0.00001)

    return np.mean(mean_ave_p, axis=0)


def mean_reciprocal_rank(correct_mat):
    first_hit = np.argmax(correct_mat, axis=1)
    first_hit = np.asarray(first_hit + 1, np.float)

    return np.mean(1.0 / first_hit)


def average_precision(correct_mat, ret_num=None):
    data = correct_mat if ret_num == None else correct_mat[:,:ret_num]
    return np.mean(data)

def recall_at_n(correct_mat, ret_num):
    recall = np.max(correct_mat[:,:ret_num],axis=1)
    return np.mean(recall)