from typing import Optional
import os
import sys


hidden_size: int = 512
dec_hidden_size: Optional[int] = 512
embed_size: int = 300

# Data
json_data_path = '../data/服饰_50k.json'
samples_path = '../data/samples.txt'
max_vocab_size = 20000
embed_file: Optional[str] = None  # use pre-trained embeddings
train_data_path: str = '../data/train.txt'
val_data_path: Optional[str] = '../data/val.txt'
test_data_path: Optional[str] = '../data/test.txt'
stop_word_file = './data/HIT_stop_words.txt'
max_src_len: int = 300  # exclusive of special tokens such as EOS
max_tgt_len: int = 100  # exclusive of special tokens such as EOS
truncate_src: bool = True
truncate_tgt: bool = True
min_dec_steps: int = 30
max_dec_steps: int = 100
enc_rnn_dropout: float = 0.5
enc_attn: bool = True
dec_attn: bool = True
pointer: bool = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0
is_cuda = True
encoder_save_name = './saved_model/encoder.pt'
decoder_save_name = './saved_model/decoder.pt'
attention_save_name = './saved_model/attention.pt'
reduce_state_save_name = './saved_model/reduce_state.pt'
losses_path = './saved_model/val_losses.pkl'
max_grad_norm = 2.0


# Training
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.001
lr_decay = 0.0
initial_accumulator_value = 0.1
epochs = 1
batch_size = 8
coverage = False
fine_tune = False
log_path = '../runs/baseline'

# Testing

# Beam search
beam_size: int = 3
alpha = 0.2
beta = 0.2
gamma = 0.1

