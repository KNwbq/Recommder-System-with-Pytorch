# -*- coding: utf-8 -*-
# @Time    : 2022/3/7 10:41
# @Author  : p1ayer
# @E-mail  : cskyun_ng@mail.scut.edu.cn

import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    file_path = "ml-1m/ratings.dat"
    data = pd.read_csv(file_path, sep="::", engine="python",
                       names=["user_id", "item_id", "label", "Timestamp"]).sort_values(by="Timestamp")
    rows, cols = data.shape
    split_index = int(rows * 0.7)
    data_train: pd.DataFrame = data.iloc[:split_index, :].sort_values(by="Timestamp")
    data_test: pd.DataFrame = data.iloc[split_index:, :].sort_values(by="Timestamp")

    data_train.to_csv("ml-1m/train.csv", header=None, index=False)
    data_test.to_csv("ml-1m/test.csv", header=None, index=False)
    print("划分完毕")