#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2020/12
# author:Yushan Zheng
# emai:yszheng@buaa.edu.cn

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from thirdparty.diffpool import SoftPoolingGcnEncoder
import numpy as np


class RegionSelfAttention(nn.Module):
    def __init__(self,seq_dim, region_dim, hidden_dim, dropout=0.0):
        super(RegionSelfAttention, self).__init__()
        self.query_layer = nn.Linear(seq_dim, hidden_dim, bias=False)
        self.key_layer = nn.Linear(region_dim, hidden_dim, bias=False)
        self.value_layer = nn.Linear(region_dim, hidden_dim, bias=False)
        self.sqrt_d = hidden_dim ** 0.5
    
    def construct_mask(self, max_sql, sq_len): 
        # masks
        bach_size = sq_len.shape[0]
        out_tensor = torch.zeros(sq_len.shape[0], max_sql)
        for i in range(bach_size):
            out_tensor[i, :sq_len[i]] = 1

        return out_tensor.unsqueeze(1).cuda()


    def forward(self, seq_code, region_features, sq_len):
        query = self.query_layer(seq_code)
        key = self.key_layer(region_features)
        if query.dim() < key.dim():
            query = self.query_layer(seq_code).unsqueeze(1)
        value = F.relu(self.value_layer(region_features))

        dist = query.matmul(key.transpose(1,2))
 
        non_zero_roi = self.construct_mask(key.shape[1], sq_len)
        attention_score = F.softmax(dist / self.sqrt_d, dim=2)

        attention_score = attention_score * non_zero_roi
        attention_score = attention_score / torch.sum(attention_score, dim=2, keepdim=True)

        output = attention_score.matmul(value) + query
        
        return output.squeeze(), attention_score.squeeze()


class DRANet(nn.Module):
    def __init__(self, num_classes, hash_bits=32, 
                    rnn_model='GRU', num_rnn_layers=1, num_rnn_hdim=128, max_sq_len=64,
                    dropout=0.0, disiable_att=False, args=None):

        super(DRANet, self).__init__()
        self.hash_bits = hash_bits
        self.max_sq_len = max_sq_len
        self.rnn_model = rnn_model
        self.self_attention = not disiable_att

        self.graph_encoder = SoftPoolingGcnEncoder(args.max_nodes, args.input_dim, 
                args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, args.assign_ratio, args.num_gc_layers, args.num_pool,
                bn=args.bn, dropout=dropout, linkpred=args.linkpred, args=args)

        graph_feature_dim = self.graph_encoder.output_dim()

        if self.rnn_model in ('RNN', 'LSTM', 'GRU'):
            self.rnn = nn.__dict__[self.rnn_model](
                input_size=graph_feature_dim, hidden_size=num_rnn_hdim, 
                num_layers=num_rnn_layers, batch_first=True
                )
            self.rnn.flatten_parameters()
        else:
            raise NotImplementedError('The RNN model is unkown')

        if self.self_attention:
            self.region_attention_layer = RegionSelfAttention(seq_dim=num_rnn_hdim,
                                            region_dim=graph_feature_dim,
                                            hidden_dim=num_rnn_hdim)

        self.pred_layer = nn.Linear(num_rnn_hdim, num_classes)
        self.hash_layer = nn.Linear(num_rnn_hdim, self.hash_bits)


    def forward(self, data):
        feats = data[0].float().view(-1,data[0].size()[-2],data[0].size()[-1]).cuda()
        adj = data[1].float().view(-1,data[1].size()[-2],data[1].size()[-1]).cuda()
        nodes = data[2].int().view(-1).cuda()
        sq_len = data[3].view(-1)

        # GCN feature
        tmp = nodes > 0
        graph_rep_ = self.graph_encoder(feats[tmp], adj[tmp], nodes[tmp], assign_x=feats[tmp])
        graph_rep = torch.zeros(nodes.size()[0], graph_rep_.size()[-1]).cuda()
        graph_rep[tmp] = graph_rep_

        # RNN
        sequence = graph_rep.reshape(-1, self.max_sq_len, graph_rep.size()[-1])
        sq_len, sorted_indices = torch.sort(sq_len, descending=True)
        sequence = sequence.index_select(0, sorted_indices.cuda())
        packed_sequence = pack_padded_sequence(sequence, sq_len, batch_first=True)

        if self.rnn_model=='RNN' or self.rnn_model=='GRU':
            _, hn = self.rnn(packed_sequence)
            hn = hn[-1,:,:]
        elif self.rnn_model == 'LSTM':
            _, (hn, _) = self.rnn(packed_sequence)
            hn = hn[-1,:,:]

        attention_score = None
        if self.self_attention:
            hn, attention_score = self.region_attention_layer(
                hn, sequence, sq_len
                )
            attention_score = attention_score.clone().detach().data
        
        hn_ = hn.clone()
        hn_[sorted_indices] = hn
        
        predict = self.pred_layer(hn_)
        hash_code = torch.tanh(self.hash_layer(hn_))

        return predict, hash_code, attention_score

    def get_hash_weights(self):
        return self.hash_layer.weight
