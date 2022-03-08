# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 19:58
# @Author  : p1ayer
# @E-mail  : cskyun_ng@mail.scut.edu.cn


import numpy as np
import torch
import torch.nn.functional as F
import random

activator_getter = {"identity": lambda x: x, "relu": F.relu, "tanh": F.tanh, "sigmoid": F.sigmoid}


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get("batch_size", 128)
    if len(tensors) == 1:
        tensors = tensors[0]
        for i in range(0, len(tensors), batch_size):
            yield tensors[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def set_seed(seed, cuda=False):
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def str2bool(v):
    return v.lower() in "true"


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def _compute_apk(targets, predictions, k):

    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not list(targets):
        return 0.0

    return score / min(len(targets), k)


def _compute_precision_recall(targets, predictions, k):

    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    return precision, recall


def evaluate_ranking(model, test, train=None, k=10):
    """
    Compute Precision@k, Recall@k scores and average precision (AP).
    One score is given for every user with interactions in the test
    set, representing the AP, Precision@k and Recall@k of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, rated items in
        interactions will be excluded.
    k: int or array of int,
        The maximum number of predicted items
    """

    test = test.to_csr()

    if train is not None:
        train = train.to_csr()

    if not isinstance(k, list):
        ks = [k]
    else:
        ks = k

    precisions = [list() for _ in range(len(ks))]
    recalls = [list() for _ in range(len(ks))]
    apks = list()

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)
        predictions = predictions.argsort()

        if train is not None:
            rated = set(train[user_id].indices)
        else:
            rated = []

        predictions = [p for p in predictions if p not in rated]

        targets = row.indices

        for i, _k in enumerate(ks):
            precision, recall = _compute_precision_recall(targets, predictions, _k)
            precisions[i].append(precision)
            recalls[i].append(recall)

        apks.append(_compute_apk(targets, predictions, k=np.inf))

    precisions = [np.array(i) for i in precisions]
    recalls = [np.array(i) for i in recalls]

    if not isinstance(k, list):
        precisions = precisions[0]
        recalls = recalls[0]

    mean_aps = np.mean(apks)

    return precisions, recalls, mean_aps