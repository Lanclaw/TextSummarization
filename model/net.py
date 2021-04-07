import config
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return h, c


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
        pass


