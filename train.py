import torch
from data_process import data_utils, dataset, files_utils, vocab
import config
from torch.utils.data import DataLoader


def train(train_dataset, val_dataset, vocab_list, start_epoch=0):

    print("loading data...")
    train_data = dataset.IdPair(train_dataset.pairs, vocab_list)
    val_data = dataset.IdPair(val_dataset.pairs, vocab_list)

    train_dataloader = DataLoader(dataset=train_data, batch_size=config.batch_size,
                                  shuffle=True, collate_fn=data_utils.collate_fn)

    for i, data in enumerate(train_dataloader):
        print(len(data))
        print(data[0])
        print(data[4])
        break


if __name__ == '__main__':
    test_pairs = dataset.WordPairs(config.train_data_path)
    vocab_list = test_pairs.build_vocab()
    train(test_pairs, test_pairs, vocab_list, start_epoch=0)

