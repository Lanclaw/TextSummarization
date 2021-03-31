from typing import Callable
import config

def simple_tokenize(sentence):
    return sentence.split()

class PairDataset(object):
    def __init__(self,
                 filename,
                 tokenizer: Callable = simple_tokenize,
                 max_src_len: int = None,
                 max_tgt_len: int = None,
                 truncate_src: bool = False,
                 truncate_tgt: bool = False):
        self.filename = filename
