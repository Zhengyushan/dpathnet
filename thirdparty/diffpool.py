
#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Modified from code for paper 
"Ying et al. Hierarchical Graph Representation Learning with Differentiable Pooling, NeurIPS 2018"
Author: RexYing (github username)
Github repo: https://github.com/RexYing/diffpool
"""

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
import time

import numpy as np

# GCN basic operation


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(
            torch.FloatTensor(output_dim)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)

        adj_ = adj + torch.eye(adj.shape[1]).cuda()
        degree_inv_sqrt = torch.eye(adj.shape[1]).expand(adj.shape[0], adj.shape[1], adj.shape[2]).cuda() \
            * (1.0/adj_.sum(dim=1, keepdim=True).sqrt())
        adj_ = torch.matmul(degree_inv_sqrt, adj_).matmul(degree_inv_sqrt)

        y = torch.matmul(adj_, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)

        return y


class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1

        self.bias = True

        if args is not None:
            self.ar = args.assign_ratio
            self.max_nodes = args.max_nodes

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim, hidden_dim, embedding_dim, num_layers,
            add_self, normalize=True, dropout=dropout)
        if self.bn:
            self.bn_first, self.bn_block = self.build_bn_layers(
                self.max_nodes, num_layers)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
                          normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                               normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                       normalize_embedding=normalize, dropout=dropout, bias=self.bias)
             for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                              normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def build_bn_layers(self, number_nodes, num_layers):
        bn_first = nn.BatchNorm1d(number_nodes)
        bn_block = nn.ModuleList([nn.BatchNorm1d(number_nodes)
                                  for i in range(num_layers - 2)])

        return bn_first, bn_block

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, bn_first, bn_block, embedding_mask=None):
        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = bn_first(x)

        x_all = [x]

        for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = bn_block[i](x)

            x_all.append(x)
        x = conv_last(x, adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def loss(self, pred, label, type='softmax'):
        # softmax + CE
        if type == 'softmax':
            return F.cross_entropy(pred, label, size_average=True)

        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(
                batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], concat=True, bn=True, dropout=0.5, linkpred=True,
                 assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                    num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, bn=bn, args=args)
        add_self = not concat
        self.concat = concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred

        # GC
        self.conv_first_after_pool = nn.ModuleList([])
        self.conv_block_after_pool = nn.ModuleList([])
        self.conv_last_after_pool = nn.ModuleList([])
        self.bn_first_modules = nn.ModuleList([])
        self.bn_block_modules = nn.ModuleList([])

        current_node = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            self.conv_first2, self.conv_block2, self.conv_last2 = self.build_conv_layers(
                self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)

            self.conv_first_after_pool.append(self.conv_first2)
            self.conv_block_after_pool.append(self.conv_block2)
            self.conv_last_after_pool.append(self.conv_last2)

            if self.bn:
                bn_first, bn_block = self.build_bn_layers(
                    current_node, num_layers)
                self.bn_first_modules.append(bn_first)
                self.bn_block_modules.append(bn_block)

            current_node = int(current_node * assign_ratio)
        # assignment
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList([])
        self.assign_conv_block_modules = nn.ModuleList([])
        self.assign_conv_last_modules = nn.ModuleList([])
        self.assign_bn_first_modules = nn.ModuleList([])
        self.assign_bn_block_modules = nn.ModuleList([])

        self.assign_pred_modules = nn.ModuleList([])
        current_node = max_num_nodes
        for i in range(num_pooling):
            if i > 0:
                assign_input_dim = self.pred_input_dim

            if self.bn:
                bn_first, bn_block = self.build_bn_layers(
                    current_node, num_layers)
                self.assign_bn_first_modules.append(bn_first)
                self.assign_bn_block_modules.append(bn_block)

            current_node = int(current_node * assign_ratio)
            self.assign_conv_first, self.assign_conv_block, self.assign_conv_last = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, current_node, assign_num_layers, add_self,
                normalize=True)
            assign_pred_input_dim = assign_hidden_dim * \
                (num_layers - 1) + current_node if concat else current_node
            self.assign_pred = self.build_pred_layers(
                assign_pred_input_dim, [], current_node, num_aggs=1).cuda()

            # next pooling layer
            assign_input_dim = embedding_dim

            self.assign_conv_first_modules.append(self.assign_conv_first)
            self.assign_conv_block_modules.append(self.assign_conv_block)
            self.assign_conv_last_modules.append(self.assign_conv_last)
            self.assign_pred_modules.append(self.assign_pred)

    def output_dim(self):
        if self.concat:
            output_dim = self.pred_input_dim * (self.num_pooling+1)
        else:
            output_dim = self.pred_input_dim

        return output_dim

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(
                max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        embedding_tensor = self.gcn_forward(x, adj,
                                            self.conv_first, self.conv_block, self.conv_last, self.bn_first, self.bn_block, embedding_mask)

        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(
                    max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            self.assign_tensor = self.gcn_forward(x_a, adj,
                                                  self.assign_conv_first_modules[i], self.assign_conv_block_modules[
                                                      i], self.assign_conv_last_modules[i],
                                                  self.assign_bn_first_modules[i], self.assign_bn_block_modules[i],
                                                  embedding_mask)
            # [batch_size x num_nodes x next_lvl_num_nodes]
            self.assign_tensor = nn.Softmax(
                dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            # update pooled features and adj matrix
            x = torch.matmul(torch.transpose(
                self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1,
                                  2) @ adj @ self.assign_tensor
            x_a = x

            embedding_tensor = self.gcn_forward(x, adj,
                                                self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                                                self.conv_last_after_pool[i], self.bn_first_modules[i], self.bn_block_modules[i],)

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                #out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out

        return output
