import torch
from torch import optim
from data_process import data_utils, dataset, files_utils, vocab
import config
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset import WordPairs, IdPair
from net import Seq2Seq
from data_utils import collate_fn
import pickle
from tqdm import tqdm
import numpy as np
import os
from torch.nn.utils import clip_grad_norm_


def train(dataset, val_dataset, v, start_epoch=0):
    """Train the model, evaluate it and store it.

    Args:
        dataset (dataset.PairDataset): The training dataset.
        val_dataset (dataset.PairDataset): The evaluation dataset.
        v (vocab.Vocab): The vocabulary built from the training dataset.
        start_epoch (int, optional): The starting epoch number. Defaults to 0.
    """
    print('loading model')
    DEVICE = torch.device("cuda" if config.is_cuda else "cpu")

    model = Seq2Seq(v)
    model.load_model()
    model.to(DEVICE)

    # forward
    print("loading data")
    train_data = IdPair(dataset.pairs, v)
    val_data = IdPair(val_dataset.pairs, v)

    print("initializing optimizer")

    # Define the optimizer.
    optimizer = optim.Adam(model.parameters(),
                           lr=config.learning_rate)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)

    val_losses = np.inf
    if (os.path.exists(config.losses_path)):
        with open(config.losses_path, 'rb') as f:
            val_losses = pickle.load(f)


    # SummaryWriter: Log writer used for TensorboardX visualization.
    # writer = SummaryWriter(config.log_path)
    # tqdm: A tool for drawing progress bars during training.
    # with tqdm(total=config.epochs) as epoch_progress:
    epochs = tqdm(range(start_epoch, config.epochs))
    for epoch in epochs:
        batch_losses = []  # Get loss of each batch.

        with tqdm(total=len(train_dataloader)) as batch_progress:
            for batch, data in enumerate(tqdm(train_dataloader)):
                x, y, x_len, y_len, oov, len_oovs = data
                assert not np.any(np.isnan(x.numpy()))
                if config.is_cuda:  # Training with GPUs.
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)

                model.train()  # Sets the module in training mode.
                optimizer.zero_grad()  # Clear gradients.
                # Calculate loss.
                loss = model(x, y)
                batch_losses.append(loss.item())
                loss.backward()  # Backpropagation.

                # Do gradient clipping to prevent gradient explosion.
                clip_grad_norm_(model.encoder.parameters(),
                                config.max_grad_norm)
                clip_grad_norm_(model.decoder.parameters(),
                                config.max_grad_norm)
                clip_grad_norm_(model.attention.parameters(),
                                config.max_grad_norm)
                optimizer.step()  # Update weights.

                # Output and record epoch loss every 100 batches.
                if (batch % 5) == 0 and batch:
                    batch_progress.set_description(f'Epoch {epoch}')
                    batch_progress.set_postfix(Batch=batch, Loss=loss.item())
                    batch_progress.update(5)
                #     # Write loss for tensorboard.
                #     writer.add_scalar(f'Average loss for epoch {epoch}',
                #                       np.mean(batch_losses),
                #                       global_step=batch)
        # Calculate average loss over all batches in an epoch.
        epoch_loss = np.mean(batch_losses)

        epochs.set_description(f'Epoch {epoch}')
        epochs.set_postfix(Loss=epoch_loss)
        epochs.update()

    # writer.close()


if __name__ == "__main__":
    # Prepare dataset for training.
    DEVICE = torch.device('cuda') if config.is_cuda else torch.device('cpu')
    train_dataset = WordPairs(config.test_data_path,
                          max_src_len=config.max_src_len,
                          max_tgt_len=config.max_tgt_len,
                          truncate_src=config.truncate_src,
                          truncate_tgt=config.truncate_tgt)
    val_dataset = WordPairs(config.val_data_path,
                              max_src_len=config.max_src_len,
                              max_tgt_len=config.max_tgt_len,
                              truncate_src=config.truncate_src,
                              truncate_tgt=config.truncate_tgt)

    vocab = train_dataset.build_vocab()

    train(train_dataset, val_dataset, vocab, start_epoch=0)

