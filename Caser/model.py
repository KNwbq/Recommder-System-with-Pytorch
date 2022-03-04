# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 23:20
# @Author  : p1ayer
# @E-mail  : cskyun_ng@mail.scut.edu.cn

import torch
from torch import nn


class Caser(nn.Module):
    def __init__(self,
                 feature_):
        super(Caser, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.args = args
