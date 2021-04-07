import copy
from collections import Counter
from typing import Callable
import config
import numpy as np
import torch
from functools import wraps


def f():
    x = [1, 2]
    y = [3, 4]
    return x, y


x = f()
print(x)




