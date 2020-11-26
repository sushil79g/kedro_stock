import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, i_size, h_size, n_layers, o_size):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size = i_size,
            hidden_size = h_size,
            num_layers = n_layers
        )

        self.out = nn.Linear(h_size, o_size)
    
    def forward(self, x, h_state):
        r_out, hidden_size = self.rnn(x, h_state)
        hidden_size = hidden_size[-1].size(-1)
        r_out = r_out.view(-1, hidden_size)
        outs = self.out(r_out)

        return outs, hidden_size