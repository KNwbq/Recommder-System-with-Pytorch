# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 19:58
# @Author  : p1ayer
# @E-mail  : cskyun_ng@mail.scut.edu.cn

# import pandas as pd
# import random
# from tqdm import tqdm
# from utils_.utils import get_logger, fix_rand
# fix_rand(2022)
#
#
# def sparse_feature(feat, feat_num, embed_dim: int = 4):
#     """
#     create dictionary for sparse_feature
#     :param feat: feature_name
#     :param feat_num: the base number of sparse features
#     :param embed_dim: embedding dimension
#     :return: A dictionary for sparse_feature
#     """
#     return {"feat": feat, "feat_num": feat_num, "embed_dim": embed_dim}
#
#
# def create_implicit_ml_1m_dataset(file, trans_score=2, embed_dim=8, maxlen=40):
#     """
#     :param file: string. dataset path
#     :param trans_score: A scalar.
#     :param embed_dim: A scalar. latent factor
#     :param maxlen: max length
#     :return: user_num, item_num, train_df, test_df
#     """
#     logger = get_logger()
#     logger.info("Data Preprocess Start")
#     df = pd.read_csv(file, sep="::", engine="python",
#                      names=["user_id", "item_id", "label", "Timestamp"])
#     df = df[df.label >= trans_score]
#     train_data, val_data, test_data = [], [], []
#     item_id_max = df["item_id"].max()
#     for user_id, df in tqdm(df[["user_id", "item_id"]].groupby("user_id")):
#         pos_list = df["item_id"].tolist()
#
#         # TODO only the item idx not in the pos_list be the neg?
#         def generate_neg():
#             neg = pos_list[0]
#             while neg in pos_list:
#                 neg = random.randint(1, item_id_max)
#             return neg
#
#         neg_list = [generate_neg() for _ in range(len(pos_list) + 100)]
#         for i in range(1, len(pos_list)):
#             hist_i = pos_list[:i]
#             while len(hist_i) < maxlen:
#                 hist_i.insert(0, 0)
#             if i == len(pos_list) - 1:
#                 test_data.append([user_id, hist_i, pos_list[i], 1])
#                 for _neg in neg_list:
#                     test_data.append([user_id, hist_i, _neg, 0])
#             elif i == len(pos_list) - 2:
#                 val_data.append([user_id, hist_i, pos_list[i], 1])
#                 val_data.append([user_id, hist_i, neg_list[i], 0])
#             else:
#                 train_data.append([user_id, hist_i, pos_list[i], 1])
#                 train_data.append([user_id, hist_i, neg_list[i], 0])
#     user_num, item_num = df["user_id"].max()+1, df["item_id"].max()+1
#     feature_columns = [sparse_feature("user_id", user_num, embed_dim),
#                        sparse_feature("item_id", item_num, embed_dim)]
#
#     random.shuffle(train_data)
#     random.shuffle(val_data)
#
#     train = pd.DataFrame(train_data, columns=["user_id", "hist", "target_item", "label"])
#     val = pd.DataFrame(val_data, columns=["user_id", "hist", "target_item", "label"])
#     test = pd.DataFrame(test_data, columns=["user_id", "hist", "target_item", "label"])
#     train_X = [train["user_id"], train["hist"], train["target_item"].values]
#     train_y = [train["label"]]
#     val_X = [val["user_id"], val["hist"], val["target_item"].values]
#     val_y = [val["label"]]
#     test_X = [test["user_id"], test["hist"], test["target_item"].values]
#     test_y = [test["label"]]
#     logger.info("Data Preprocess End")
#     return feature_columns, train_X, train_y, val_X, val_y, test_X, test_y
#
#
# if __name__ == "__main__":
#     create_implicit_ml_1m_dataset("../ml-1m/ratings.dat")

import numpy as np
import torch
import torch.nn.functional as F
import random

activator_getter = {"identity": lambda x: x, "relu": F.relu, "tanh": F.tanh, "sigmoid": F.sigmoid}


def set_seed(seed, cuda=False):
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def str2bool(v):
    return v.lower() in "true"
