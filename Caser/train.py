# -*- coding: utf-8 -*-
# @Time    : 2022/3/5 19:35
# @Author  : p1ayer
# @E-mail  : cskyun_ng@mail.scut.edu.cn
import torch
from model import Caser
from torch import optim


class Recommender(object):
    def __init__(self,
                 n_iter: int = 1,
                 batch_size: int = 64,
                 l2: float = 0.0,
                 neg_samples: int = 1,
                 learning_rate: float = 0.01,
                 use_cuda: bool = False,
                 model_args=None):

        # model related
        self._num_items = None
        self._num_users = None
        self._net = None
        self.model_args = model_args

        # learning related
        self._batch_size = batch_size
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._neg_samples = neg_samples
        self._device = torch.device("cuda" if use_cuda else "cpu")

        # rank evaluate related
        self.test_sequence = None
        self._candidate = dict()

    @property
    def _initialized(self):
        return self._net is None

    def _initialize(self, interactions):
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self.test_sequence = interactions.test_sequences

        self._net = Caser(self._num_users,
                          self._num_items,
                          self.model_args)

        self._optimizer = optim.Adam(self._net.parameters(),
                                     weight_decay=self._l2,
                                     lr=self._learning_rate)

    def fit(self, train, test, verbose=False):

