# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 23:20
# @Author  : p1ayer
# @E-mail  : cskyun_ng@mail.scut.edu.cn

import torch
from torch import nn
from utils import activator_getter


class Caser(nn.Module):
    def __init__(self, num_users, num_items, model_args):
        """
        :param num_users: number of user in the system
        :param num_items: number of item in the system
        :param model_args: the parameters of the model, consist of window size L,
        the number of the horizon filter, the number of the vertical filter and the dropout ratio
        """
        super(Caser, self).__init__()
        self.args = model_args

        L = self.args.L
        embed_dim = self.args.d
        self.nh = self.args.nh
        self.nv = self.args.nv
        self.drop_ratio = self.args.drop

        self.ac_conv = activator_getter[self.args.ac_conv]
        self.ac_fc = activator_getter[self.args.ac_fc]

        # embedding
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # vertical convolution
        self.vertical = nn.Conv2d(1, self.nv, (L, 1))

        # horizon convolution
        lengths = [(i+1) for i in range(L) for _ in range(self.nh)]
        lengths = sorted(lengths, key=lambda x: x)
        self.horizon = nn.ModuleList([nn.Conv2d(1, 1, (i, embed_dim)) for i in lengths])

        # fully-connect1
        self.fc1_v_dim = self.nv * embed_dim
        self.fc1_h_dim = sum([L - i for i in lengths])
        self.fc1_dim = self.fc1_v_dim + self.fc1_h_dim
        self.fc1 = nn.Linear(self.fc1_dim, embed_dim)

        # fully-connect2
        self.W2 = nn.Embedding(num_items, embed_dim * 2)
        self.b2 = nn.Embedding(num_items, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Embedding, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, seq, user, items, _predict):
        """
        :param seq: (batch_size, max_length)
        :param user: (batch_size, 1)
        :param items: the desired item; (batch_size, len)
        :param _predict: predict or not
        :return: the probability of each items that will be chosen
        """
        emb4item = self.item_embedding(seq)
        emb4user = self.user_embedding(user)

        ver_res = self.vertical(emb4item)
        hor_res = []
        for conv in self.horizon:
            hor_res.append(self.ac_fc(conv(emb4item)))

        ver_hor = torch.cat((ver_res, torch.cat(hor_res, 0)), 0)
        ver_hor_user = torch.cat((ver_hor, emb4user), 0)

        ver_hor_user_fc1 = self.ac_fc(self.fc1(ver_hor_user))

        w2 = self.W2(items)
        b2 = self.b2(items)

        res = (ver_hor_user_fc1 * w2).sum(1) + b2
        return res








