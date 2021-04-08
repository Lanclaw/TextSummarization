import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, drop: float = 0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, dropout=drop, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded)    # output: [b_s, seq_len, h_s * 2]  hidden: [2, 2 * layers, b_s, h_s]

        return output, hidden


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        h, c = hidden
        h_reduced = torch.sum(h, dim=0, keepdim=True)   # [1, b_s, h_s]
        c_reduced = torch.sum(c, dim=0, keepdim=True)   # [1, b_s, h_s]
        return h_reduced, c_reduced


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wh = nn.Linear(2*hidden_size, 2*hidden_size, bias=False)
        self.Ws = nn.Linear(2*hidden_size, 2*hidden_size)
        self.v = nn.Linear(2*hidden_size, 1, bias=False)

    def forward(self, decoder_states, encoder_output, x_padding_masks):
        h_dec, c_dec = decoder_states
        s_t = torch.cat([h_dec, c_dec], dim=2)  # [1, b_s, 2*h_s]
        s_t = s_t.transpose(0, 1)   # [b_s, 1, 2*h_s]
        s_t = s_t.expand_as(encoder_output).contiguous()    # [b_s, seq_len, 2*h_s]

        encoder_features = self.Wh(encoder_output.contiguous())     # [b_s, seq_len, 2*h_s]
        decoder_features = self.Ws(s_t)
        att_inputs = encoder_features + decoder_features
        score = self.v(torch.tanh(att_inputs))  # [b_s, seq_len, 1]
        att_weights = F.softmax(score, dim=1).squeeze(2)    # [b_s, seq_len]
        att_weights = att_weights * x_padding_masks
        normalization_factor = att_weights.sum(1, keepdim=True)     # [b_s, 1]
        att_weights = att_weights / normalization_factor
        context_vector = torch.bmm(att_weights.unsqueeze(1), encoder_output)    # b_s: [1, seq_len] * [seq_len, 2*h_s]
        context_vector = context_vector.squeeze(1)  # [b_s, 2*h_s]

        return context_vector, att_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, is_cuda=True):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.DEVICE = torch.device('cuda') if is_cuda else torch.device('cpu')
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.W1 = nn.Linear(hidden_size*3, hidden_size)
        self.W2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, decoder_input, decoder_states, context_vector):
        decoder_emb = self.embedding(decoder_input)     # [b_s, 1, h_s]
        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)     # [b_s, 1, h_s], [1, b_s, h_s]
        decoder_output = decoder_output.view(-1, self.hidden_size)
        concat_vector = torch.cat([decoder_output, context_vector], dim=-1)     # [b_s, h_s*3]

        ff1_out = self.W1(concat_vector)
        ff2_out = self.W2(ff1_out)  # [b_s, vocab_size]
        p_vocab = F.softmax(ff2_out, dim=1)

        return p_vocab, decoder_states


class Seq2Seq(nn.Module):
    def __init__(self, v):
        super(Seq2Seq, self).__init__()
        self.v = v
        self.DEVICE = torch.device('cuda') if config.is_cuda is True else torch.device('cpu')
        self.encoder = Encoder(len(v), config.embed_size, config.hidden_size)
        self.reduce_state = ReduceState()
        self.attention = Attention(config.hidden_size)
        self.decoder = Decoder(len(v), config.embed_size, config.hidden_size)

    def load_model(self):
        if os.path.exists(config.encoder_save_name):
            self.encoder = torch.load(config.encoder_save_name)
            self.reduce_state = torch.load(config.reduce_state_save_name)
            self.attention = torch.load(config.attention_save_name)
            self.decoder = torch.load(config.decoder_save_name)
            return True
        return False

    def forward(self, x, y):
        oov_token = torch.full(x.shape, self.v.UNK).long().to(self.DEVICE)
        x_copy = torch.where(x > len(self.v) - 1, oov_token, x)
        x_padding_masks = torch.ne(x_copy, 0).byte().float()
        encoder_output, encoder_states = self.encoder(x_copy)
        decoder_states = self.reduce_state(encoder_states)

        time_step_losses = []
        for t in range(y.shape[-1]-1):
            decoder_input_t = y[:, t]   # [b_s]
            decoder_target_t = y[:, t+1]    # [b_s]
            context_vector, att_weights = self.attention(decoder_states, encoder_output, x_padding_masks)
            p_vocab, decoder_states = self.decoder(decoder_input_t.unsqueeze(1), decoder_states, context_vector)

            target_probs = torch.gather(p_vocab, 1, decoder_target_t.unsqueeze(1))  # [b_s, 1]
            loss = -torch.log(target_probs + config.eps)
            mask = torch.ne(decoder_target_t, 0).byte().float()
            loss = loss * mask
            time_step_losses.append(loss)

        # time_step_losses: [y_len, b_s]
        batch_losses = torch.stack(time_step_losses, 1)     # [b_s, y_len]
        batch_losses = torch.sum(batch_losses, 1)   # [b_s]

        seq_len_mask = torch.ne(y, 0).byte().float()    # [b_s, y_len]
        batch_seq_len = torch.sum(seq_len_mask, 1)  # [b_s]

        batch_loss = torch.mean(batch_losses / batch_seq_len)
        return batch_loss


if __name__ == '__main__':
    print(1)