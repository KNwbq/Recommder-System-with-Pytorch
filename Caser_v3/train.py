# -*- coding: utf-8 -*-
# @Time    : 2022/3/5 19:35
# @Author  : p1ayer
# @E-mail  : cskyun_ng@mail.scut.edu.cn

import argparse
from time import time
from sklearn.utils import shuffle

from model import Caser
from torch import optim
from utils import *
from interactions import Interactions
from tqdm import tqdm


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

        # rank evaluation related
        self.test_sequence = None
        self._candidate = dict()

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):
        self._num_items = interactions.num_item
        self._num_users = interactions.num_user

        self.test_sequence = interactions.test_sequences

        self._net = Caser(self._num_items,
                          self.model_args).to(self._device)

        self._optimizer = optim.Adam(self._net.parameters(),
                                     weight_decay=self._l2,
                                     lr=self._learning_rate)

    def fit(self, train, test, verbose=False):
        """
        The general training loop to fit the model

        Parameters
        ----------

        train: :class:`spotlight.interactions.Interactions`
            training instances, also contains test sequences
        test: :class:`spotlight.interactions.Interactions`
            only contains targets for test sequences
        verbose: bool, optional
            print the logs
        """

        # convert to sequences, targets and users
        sequences_np = train.sequences.sequences
        targets_np = train.sequences.target
        users_np = train.sequences.user_id.reshape(-1, 1)
        item4user_np = train.sequences.item4user

        n_train = sequences_np.shape[0]

        output_str = 'total training instances: %d' % n_train
        print(output_str)

        if not self._initialized:
            self._initialize(train)

        start_epoch = 0

        for epoch_num in range(start_epoch, self._n_iter):
            t1 = time()

            # set model to training mode
            self._net.train()
            users_np, sequences_np, targets_np, item4user_np = shuffle(users_np, sequences_np, targets_np, item4user_np)
            neg_samples = self._generate_negative_samples(users_np, train, self._neg_samples)
            users, sequences, targets, negatives, item4user = (torch.from_numpy(users_np).to(self._device),
                                                               torch.from_numpy(sequences_np).to(self._device),
                                                               torch.from_numpy(targets_np).to(self._device),
                                                               torch.from_numpy(neg_samples).to(self._device),
                                                               [torch.from_numpy(_).to(self._device) for _ in item4user_np])

            epoch_loss = 0.0
            minibatch_num = 0
            for (minibatch_num,
                 (batch_users,
                  batch_sequences,
                  batch_targets,
                  batch_negatives,
                  batch_item4user)) in enumerate(tqdm(minibatch(users, sequences, targets, negatives, item4user, batch_size=self._batch_size))):

                items2predict = torch.cat((batch_targets, batch_negatives), 1)
                # print(targets.size(), negatives.size())
                # print(batch_sequences.size(),
                #       batch_users.size(),
                #       items2predict.size())
                predictions = self._net(batch_sequences, items2predict, batch_item4user)
                (t_pred, n_pred) = torch.split(predictions, [batch_targets.size(1), batch_negatives.size(1)], 1)
                self._optimizer.zero_grad()
                pos_loss = -torch.mean(torch.log(torch.sigmoid(t_pred)))
                neg_loss = -torch.mean(torch.log(1 - torch.sigmoid(n_pred)))
                loss = pos_loss + neg_loss
                epoch_loss += loss.item()
                loss.backward()
                self._optimizer.step()

            epoch_loss /= (minibatch_num + 1)

            t2 = time()
            if verbose and (epoch_num + 1) % 10 == 0:
                precision, recall, mean_aps = evaluate_ranking(self, test, train, k=[1, 5, 10])
                output_str = "Epoch %d [%.1f s]\t loss=%.4f, map=%.4f, " \
                             "prec@1=%.4f, prec@5=%.4f, prec@10=%.4f, " \
                             "recall@1=%.4f, recall@5=%.4f, recall@10=%.4f, [%.1f s]" % (epoch_num + 1,
                                                                                         t2 - t1,
                                                                                         epoch_loss,
                                                                                         mean_aps,
                                                                                         np.mean(precision[0]),
                                                                                         np.mean(precision[1]),
                                                                                         np.mean(precision[2]),
                                                                                         np.mean(recall[0]),
                                                                                         np.mean(recall[1]),
                                                                                         np.mean(recall[2]),
                                                                                         time() - t2)
                print(output_str)
            else:
                output_str = "Epoch %d [%.1f s]\t loss=%.4f [%.1f s]" % (epoch_num + 1, t2 - t1,
                                                                         epoch_loss, time() - t2)
                print(output_str)

    def predict(self, user_id, item_ids=None):
        """
        Make predictions for evaluation: given a user id, it will
        first retrieve the test sequence associated with that user
        and compute the recommendation scores for items.

        Parameters
        ----------

        user_id: int
           users id for which prediction scores needed.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.
        """

        if self.test_sequence is None:
            raise ValueError('Missing test sequences, cannot make predictions')

        # set model to evaluation model
        self._net.eval()
        with torch.no_grad():
            sequences_np = self.test_sequence.sequences[user_id, :]
            sequences_np = np.atleast_2d(sequences_np)
            item4user_np = self.test_sequence.item4user[user_id]

            if item_ids is None:
                item_ids = np.arange(self._num_items).reshape(-1, 1)

            sequences = torch.from_numpy(sequences_np).long()
            item_ids = torch.from_numpy(item_ids).long()
            user_id = torch.from_numpy(np.array([[user_id]])).long()
            item4user = torch.from_numpy(item4user_np).long()

            user, sequences, items, item4user = (user_id.to(self._device),
                                                 sequences.to(self._device),
                                                 item_ids.to(self._device),
                                                 item4user.to(self._device))

            out = self._net(sequences,
                            items,
                            [item4user])

        return out.cpu().numpy().flatten()

    def _generate_negative_samples(self, users, interactions, n):
        users_ = users.squeeze()
        negative_samples = np.zeros((users_.shape[0], n), np.int64)
        if not self._candidate:
            all_items = np.arange(interactions.num_item - 1) + 1  # 0 for padding
            train = interactions.to_csr()
            for user, row in enumerate(train):
                self._candidate[user] = list(set(all_items) - set(row.indices))

        for i, u in enumerate(users_):
            for j in range(n):
                can = self._candidate[u]
                negative_samples[i, j] = can[np.random.randint(len(can))]

        return negative_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--train_root', type=str, default='../ml-1m/train.txt')
    parser.add_argument('--test_root', type=str, default='../ml-1m/test.txt')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    # train arguments
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=str2bool, default=True)

    config = parser.parse_args()

    # model dependent arguments
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--d', type=int, default=50)
    model_parser.add_argument('--nv', type=int, default=4)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')

    model_config = model_parser.parse_args()
    model_config.L = config.L

    # set seed
    set_seed(config.seed, cuda=config.use_cuda)

    # load dataset
    _train = Interactions(config.train_root)
    # transform triplets to sequence representation
    _train.to_seq(config.L, config.T)

    _test = Interactions(config.test_root,
                         user2id=_train.user2id,
                         item2id=_train.item2id)

    print(config)
    print(model_config)
    # fit model
    model = Recommender(n_iter=config.n_iter,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate,
                        l2=config.l2,
                        neg_samples=config.neg_samples,
                        model_args=model_config,
                        use_cuda=config.use_cuda)

    model.fit(_train, _test, verbose=True)