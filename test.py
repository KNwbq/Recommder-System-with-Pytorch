# -*- coding: utf-8 -*-
# @Time    : 2022/4/3 13:41
# @Author  : p1ayer
# @E-mail  : cskyun_ng@mail.scut.edu.cn

import torch

matrix = torch.randn(3, 6)
matrix_split = torch.split(matrix, [3, 3], dim=1)
print(matrix)
print(matrix_split)