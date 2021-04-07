from typing import Callable
import config
from collections import Counter
import vocab
from data_utils import simple_tokenize, count_words, src2id
from torch.utils.data import Dataset


class WordPairs(object):
    def __init__(self,
                 filename,
                 tokenizer: Callable = simple_tokenize,
                 max_src_len: int = None,
                 max_tgt_len: int = None,
                 truncate_src: bool = False,
                 truncate_tgt: bool = False):
        self.filename = filename
        self.pairs = []

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                pair = line.strip().split('<sep>')
                src = tokenizer(pair[0])
                tgt = tokenizer(pair[1])

                if max_src_len and len(src) > max_src_len:
                    if truncate_src:
                        src = src[:max_src_len]
                    else:
                        continue
                if max_tgt_len and len(tgt) > max_tgt_len:
                    if truncate_tgt:
                        tgt = tgt[:max_tgt_len]
                    else:
                        continue

                self.pairs.append((src, tgt))
        print('%d pairs read.' % len(self.pairs))

    def build_vocab(self):
        word_counts = Counter()
        count_words(word_counts, [src + tgt for src, tgt in self.pairs])

        vocab_list = vocab.Vocab()
        for word, count in word_counts.most_common(config.max_vocab_size):
            vocab_list.add_words([word])
        return vocab_list


class IdPair(Dataset):
    def __init__(self, word_pairs, vocab):
        self.src_sents = [x[0] for x in word_pairs]
        self.tgt_sents = [x[1] for x in word_pairs]
        self.vocab = vocab
        self.len = len(word_pairs)

    def __getitem__(self, index):
        x, oov = src2id(self.src_sents[index], self.vocab)
        return {
            'x': [self.vocab.SOS] + x + [self.vocab.EOS],
            'oov': oov,
            'y': [self.vocab.SOS] + [self.vocab[y] for y in self.tgt_sents[index]] + [self.vocab.EOS],
            'x_len': len(self.src_sents[index]),
            'oov_len': len(oov),
            'y_len': len(self.tgt_sents[index])
        }

    def __len__(self):
        return self.len


if __name__ == '__main__':
    train_pairs = WordPairs(config.train_data_path)
    v = train_pairs.build_vocab()
    ret = src2id(train_pairs.pairs[35][0], v)


