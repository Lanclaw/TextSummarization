import numpy as np
import torch


def simple_tokenize(sentence):
    return sentence.split()


def count_words(word_counts, text):
    for line in text:
        word_counts.update(line)


def src2id(source_words, vocab):    # based on PGN
    ids = []
    oovs = []
    unk_id = ['<UNK>']
    for w in source_words:
        id = vocab[w]
        if id == unk_id:
            if id not in oovs:              # maybe 2 same oov words
                oovs.append(w)
            oov_index = oovs.index(w)
            ids.append(vocab.size() + oov_index)
        else:
            ids.append(id)

    return ids, oovs


def sort_batch_by_len(data_batch):
    res = {
        'x': [],
        'oov': [],
        'y': [],
        'x_len': [],
        'oov_len': [],
        'y_len': []
    }
    for item in data_batch:
        res['x'].append(item['x'])
        res['oov'].append(item['oov'])
        res['y'].append(item['y'])
        res['x_len'].append(item['x_len'])
        res['oov_len'].append(item['oov_len'])
        res['y_len'].append(item['y_len'])

    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()

    data_batch = {
        name: [tensor[i] for i in sorted_indices]
        for name, tensor in res.items()
    }

    return data_batch


def collate_fn(data_batch):

    def padding(sentences, max_len, pad_idx=0):
        padded_sens = [[s] + [pad_idx] * max(0, max_len - len(s)) for s in sentences]
        return torch.tensor(padded_sens)

    sorted_batch = sort_batch_by_len(data_batch)

    x = sorted_batch['x']
    x_max_len = max([len(s) for s in x])
    y = sorted_batch['y']
    y_max_len = max([len(s) for s in y])

    x_padded = padding(x, x_max_len)
    y_padded = padding(y, y_max_len)
    x_len = torch.tensor(data_batch['x_len'])
    y_len = torch.tensor(data_batch['y_len'])

    oov = torch.tensor(data_batch['oov'])
    oov_len = torch.tensor(data_batch['oov_len'])

    return x_padded, y_padded, x_len, y_len, oov, oov_len

