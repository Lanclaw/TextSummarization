import copy
from collections import Counter
from typing import Callable
import config
import numpy as np

res = {'x': [1],
           'y': [],
           'x_len': [],
           'y_len': [],
           'OOV': [],
           'len_OOV': []}
res = res.items()
x = np.array([7, 5, 8])
y = x.argsort()[::-1]

print(res)

for name, tensor in res:
    print(name)




