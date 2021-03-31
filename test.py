import copy
from collections import Counter
from typing import Callable
import config

def f(a:Callable = None) -> int:
    return a


print(f('abc'))