# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 20:36
# @Author  : p1ayer
# @E-mail  : cskyun_ng@mail.scut.edu.cn

import logging
import random
import numpy as np
import torch


def get_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("wbqLog")
    return logger


def fix_rand(seed: int = 2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



