# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 23:20
# @Author  : p1ayer
# @E-mail  : cskyun_ng@mail.scut.edu.cn

import torch
from torch import nn
from utils import activator_getter
import torch.nn.functional as F


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
        cnt = self.nh // L
        lengths = [(i+1) for _ in range(cnt) for i in range(L)]
        lengths.extend([_ + 1 for _ in range(self.nh - len(lengths))])
        lengths = sorted(lengths, key=lambda x: x)
        self.horizon = nn.ModuleList([nn.Conv2d(1, 1, (i, embed_dim)) for i in lengths])

        # fully-connect1
        self.fc1_v_dim = self.nv * embed_dim
        self.fc1_h_dim = self.nh
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

    def forward(self, seq, user, items, _predict=True):
        """
        :param seq: (batch_size, max_length)
        :param user: (batch_size, 1)
        :param items: the desired item; (batch_size, len)
        :param _predict: predict or not
        :return: the probability of each items that will be chosen
        """
        emb4item = self.item_embedding(seq)  # (batch_size, H, W): (batch_size, L, embed_dim)
        emb4user = self.user_embedding(user)  # (batch_size, H, W): (batch_size, 1, embed_dim)

        emb4item = torch.unsqueeze(emb4item, 1)  # (batch_size, 1, L, embed_dim)
        emb4user = torch.squeeze(emb4user, 1)  # (batch_size, embed_dim)

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
        w2 = self.W2(items)
        b2 = self.b2(items)
        res = (w2 @ z_user) + b2
        return res


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
    net = Caser(ds.num_user, ds.num_item, model_config)
    _ = net(test_seq, test_user, test_item, True)