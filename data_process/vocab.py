import config
import files_utils

from collections import Counter
import numpy as np


class Vocab(object):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self):
        self.word2index = {}
        self.word_count = Counter()
        self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.index2word = self.reserved[:]
        self.embeddings = None

    def __getitem__(self, item):
        if type(item) == int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)

    def add_words(self, words):
        for word in words:
            if word not in self.index2word:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
            self.word_count.update(word)


if __name__ == '__main__':
    samples_train = files_utils.read_samples(config.train_data_path)
    samples_val = files_utils.read_samples(config.val_data_path)
    vocab = Vocab()
    for line in samples_train[:50]:
        vocab.add_words(line.strip().split())
    for line in samples_val[:50]:
        vocab.add_words(line.strip().split())


