import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, hidden_dim, vocab_size, tagset_size, n_layers):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(vocab_size, hidden_dim, n_layers)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        lstm_out, (ht, hc) = self.lstm(x)
        lstm_out = pad_packed_sequence(lstm_out)
        tag_space = self.hidden2tag(lstm_out[0])
        tag_space = tag_space[tag_space.shape[0]-1]

        output = self.softmax(tag_space)
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_dim)