# -*- coding: utf-8 -*-
# @Time    : 2022/3/6 22:50
# @Author  : p1ayer
# @E-mail  : cskyun_ng@mail.scut.edu.cn

"""
Classed describing datasets of user-item interactions
"""

import numpy as np
import scipy.sparse as sp


class Interactions(object):
    def __init__(self,
                 file_path: str = None,
                 user2id: dict = None,
                 item2id: dict = None):
        if not user2id and not item2id:
            user2id = dict()
            item2id = dict()

            num_user = 0
            num_item = 1  # item has 0 item for padding
        else:
            num_user = len(user2id)
            num_item = len(item2id)

        user_id = list()
        item_id = list()

        with open(file_path, "r") as file:
            for line in file:
                user_id.append(line.strip().split(" ")[0])
                item_id.append(line.strip().split(" ")[1])

        # for testing
        # user_id = user_id[:1000]
        # item_id = item_id[:1000]

        # df = pd.read_csv(file_path, sep=",", engine="python",
        #                  names=["user_id", "item_id", "rating", "timestamp"])
        # user_id = df["user_id"].tolist()[:10000]
        # item_id = df["item_id"].tolist()[:10000]
        # del df

        # in order to compress the user embedding
        for u in user_id:
            if u not in user2id:
                user2id[u] = num_user
                num_user += 1

        # in order to compress the item embedding. What's more, the start of the item_id is 1, cause 0 is used to pad
        for u in item_id:
            if u not in item2id:
                item2id[u] = num_item
                num_item += 1

        user_id = np.array([user2id[u] for u in user_id])
        item_id = np.array([item2id[i] for i in item_id])

        self.num_user = num_user
        self.num_item = num_item
        self.user_id = user_id
        self.item_id = item_id
        self.user2id = user2id
        self.item2id = item2id

        self.sequences = None
        self.test_sequences = None  # used to predict

    def __len__(self):
        return len(self.user_id)

    def to_coo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_id
        col = self.item_id
        data = np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_user, self.num_item))

    def to_csr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        :return the user_item matrix
        """
        return self.to_coo().tocsr()

    def to_seq(self,
               sequence_length: int = 5,
               target_length: int = 1):
        """
        Transform to sequence form.
        Valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [1, 2, 3, 4, 5, 6, 7, 8, 9], the
        returned interactions matrix at sequence length 5 and target length 3
        will be be given by:
        sequences:
           [[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]]
        targets:
           [[6, 7],
            [7, 8],
            [8, 9]]
        sequence for test (the last 'sequence_length' items of each user's sequence):
        [[5, 6, 7, 8, 9]]
        :param sequence_length: the length of the item seq, and if the length of the seq is shorted than that,
        the seq will be left-padded with zeros.
        :param target_length: seq target length.
        :return: above; list
        """
        max_seq = sequence_length + target_length

        sorted_indices = np.lexsort((self.user_id,))
        user_id = self.user_id[sorted_indices]
        item_id = self.item_id[sorted_indices]

        user_id, indices, counts = np.unique(user_id, return_index=True, return_counts=True)
        num_subsequence = sum([c - max_seq + 1 if c >= max_seq else 1 for c in counts])

        sequences = np.zeros((num_subsequence, sequence_length), dtype=np.int64)
        sequences_targets = np.zeros((num_subsequence, target_length), dtype=np.int64)
        sequences_users = np.empty(num_subsequence, dtype=np.int64)
        sequences_item4user = []

        test_sequences = np.zeros((self.num_user, sequence_length), dtype=np.int64)
        test_users = np.empty(self.num_user, dtype=np.int64)
        test_item4user = {}

        _uid = None
        for i, (uid, item_seq, item_clicked) in enumerate(_generate_sequences(user_id, item_id, indices, max_seq)):
            if uid != _uid:
                test_sequences[uid][:] = item_seq[-sequence_length:]
                test_users[uid] = uid
                _uid = uid
                test_item4user[uid] = item_clicked
            sequences[i][:] = item_seq[:sequence_length]
            sequences_users[i] = uid
            sequences_targets[i][:] = item_seq[-target_length:]
            sequences_item4user.append(item_clicked[:-target_length])

        self.sequences = SequenceInteractions(sequences_users, sequences, sequences_targets, sequences_item4user)
        self.test_sequences = SequenceInteractions(test_users, test_sequences, item4user=test_item4user)


class SequenceInteractions(object):
    def __init__(self,
                 user_id,
                 sequences,
                 targets=None,
                 item4user=None):
        self.user_id = user_id
        self.sequences = sequences
        self.target = targets
        self.item4user = item4user

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]


def _sliding_window(tensor, window_size, step_size=1):
    if len(tensor) - window_size >= 0:
        for i in range(len(tensor), 0, -step_size):
            if i - window_size >= 0:
                yield tensor[i-window_size:i], tensor[:i]
    else:
        num_padding = window_size - len(tensor)
        yield np.pad(tensor, (num_padding, 0), "constant"), tensor[:]


def _generate_sequences(user_id, item_id, indices, max_sequence_length):
    for i in range(len(indices)):
        userId = user_id[i]
        start_idx = indices[i]
        if i + 1 < len(indices):
            stop_idx = indices[i+1]
        else:
            stop_idx = None

        for seq, item4user in _sliding_window(item_id[start_idx:stop_idx], max_sequence_length):
            yield userId, seq, np.array(item4user)


if __name__ == "__main__":
    _file_path = "../ml-1m/train.txt"
    dataset = Interactions(_file_path)
    dataset.to_seq()
    print(dataset.sequences.user_id.shape,
          dataset.sequences.sequences.shape,
          dataset.sequences.target.shape,
          dataset.test_sequences.user_id.shape,
          dataset.test_sequences.sequences.shape)