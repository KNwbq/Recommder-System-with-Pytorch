# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 23:20
# @Author  : p1ayer
# @E-mail  : cskyun_ng@mail.scut.edu.cn

import torch
from torch import nn
from utils import activator_getter
import torch.nn.functional as F
import numpy as np


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x.unsqueeze(-1)).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Caser(nn.Module):
    def __init__(self, num_items, model_args):
        """
        :param num_items: number of item in the system
        :param model_args: the parameters of the model, consist of window size L,
        the number of the horizon filter, the number of the vertical filter and the dropout ratio
        """
        super(Caser, self).__init__()
        self.args = model_args

        L = self.args.L
        self.embed_dim = self.args.d
        self.nh = self.args.nh
        self.nv = self.args.nv
        self.drop_ratio = self.args.drop

        self.ac_conv = activator_getter[self.args.ac_conv]
        self.ac_fc = activator_getter[self.args.ac_fc]

        # embedding
        self.item_embedding = nn.Embedding(num_items, self.embed_dim)
        # cold start
        self.avg_user_embedding = nn.Parameter(torch.randn(self.embed_dim, dtype=torch.float32, requires_grad=True))

        # vertical convolution
        self.vertical = nn.Conv2d(1, self.nv, (L, 1))

        # horizon convolution
        lengths = [_+1 for _ in range(L)]
        self.horizon = nn.ModuleList([nn.Conv2d(1, self.nh, (i, self.embed_dim)) for i in lengths])

        # fully-connect1
        self.fc1_v_dim = self.nv * self.embed_dim
        self.fc1_h_dim = self.nh * len(lengths)
        self.fc1_dim = self.fc1_v_dim + self.fc1_h_dim
        self.fc1 = nn.Linear(self.fc1_dim, self.embed_dim)

        # fully-connect2
        self.W2 = nn.Embedding(num_items, self.embed_dim * 2)
        self.b2 = nn.Embedding(num_items, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # SEnet
        self.se = SELayer(L, 2)

        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Embedding, nn.Linear)):
        #         nn.init.xavier_uniform_(m.weight)
        # self.user_embedding.weight.data.normal_(0, 1.0 / self.user_embedding.embedding_dim)
        self.item_embedding.weight.data.normal_(0, 1.0 / self.item_embedding.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self, seq, items, item4user, _predict=True):
        """
        :param seq: (batch_size, max_length)
        :param items: the desired item; (batch_size, len)
        :param item4user: a dictionary that contains records of user
        :param _predict: predict or not
        :return: the probability of each items that will be chosen
        """
        emb4item = self.item_embedding(seq)  # (batch_size, H, W): (batch_size, L, embed_dim)
        emb4item = self.se(emb4item)
        # emb4user = torch.cat([torch.mean(self.item_embedding(torch.from_numpy(item4user[u.item()]).to(device)), dim=0) for u in user], dim=0).view(-1, emb4item.size(-1))
        emb4user = torch.cat([torch.mean(self.item_embedding(i), dim=0) for i in item4user], dim=0).view(-1, self.embed_dim) + self.avg_user_embedding
        # emb4user = self.user_embedding(user)  # (batch_size, H, W): (batch_size, 1, embed_dim)

        emb4item = torch.unsqueeze(emb4item, 1)  # (batch_size, 1, L, embed_dim)
        # emb4user = torch.squeeze(emb4user, 1)  # (batch_size, embed_dim)

        ver_res = self.vertical(emb4item)  # (batch_size, nv, 1, embed_dim)
        ver_res = ver_res.view(ver_res.size(0), -1)
        hor_res = []
        for conv in self.horizon:
            # (batch_size, 1, L-i+1)
            _hor = self.ac_conv(conv(emb4item)).squeeze(3)
            _hor = F.max_pool1d(_hor, _hor.size(2)).squeeze(2)
            hor_res.append(_hor)

        ver_hor = torch.cat((ver_res, torch.cat(hor_res, 1)), 1)
        ver_hor = self.dropout(ver_hor)
        z = self.ac_fc(self.fc1(ver_hor))
        # (batch_size, 2 * embed_dim, 1)
        z_user = torch.cat((z, emb4user), 1).unsqueeze(2)

        # (batch_size, len, 2 * embed_dim)
        # w2 = self.W2(items)
        # b2 = self.b2(items)
        # res = (w2 @ z_user) + b2

        w2 = self.W2(items)
        b2 = self.b2(items)
        # tag tag
        res = (w2 @ z_user) + b2

        # w2 = self.W2(items)
        # b2 = self.b2(items)
        # if _predict:
        #     w2 = w2.squeeze()
        #     b2 = b2.squeeze()
        #     res = (z_user * w2).sum(1) + b2
        # else:
        #     res = torch.baddbmm(b2, w2, z_user.unsqueeze(2)).squeeze()
        return res.squeeze(-1)


if __name__ == "__main__":
    from interactions import Interactions
    import argparse
    file_path = "../ml-1m/ratings.dat"
    ds = Interactions(file_path)
    ds.to_seq()
    test_seq = torch.from_numpy(ds.sequences.sequences[:100, :])
    test_user = torch.from_numpy(ds.sequences.user_id[:100])
    test_item = torch.from_numpy(ds.sequences.sequences[:100, :])

    model_parser = argparse.ArgumentParser()
    model_parser.add_argument("--d", type=int, default=50)
    model_parser.add_argument('--nv', type=int, default=4)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')
    model_parser.add_argument("--L", type=int, default=5)
    model_config = model_parser.parse_args()
    net = Caser(ds.num_user, model_config)
    _ = net(test_seq, test_user, test_item, True)