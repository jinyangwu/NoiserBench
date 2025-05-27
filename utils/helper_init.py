#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import random
import numpy as np


def set_seed(seed_value, random_lib=False, numpy_lib=False, torch_lib=False):
    '''Set random seed in the available library random, numpy and torch(cpu & cuda)'''
    if random_lib:
        random.seed(seed_value)
    if numpy_lib:
        np.random.seed(seed_value)
    if torch_lib:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
            