import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, i_size, h_size, n_layers, o_size):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=i_size,
            hidden_size=h_size,
            num_layers=n_layers
        )
        self.out = nn.Linear(h_size, o_size)

    def forward(self, x, h_state):
        r_out, hidden_state = self.rnn(x, h_state)
        
        hidden_size = hidden_state[-1].size(-1)
        r_out = r_out.view(-1, hidden_size)
        outs = self.out(r_out)

        return outs, hidden_state

def train_mlmodel(x_train, y_train, parameters):
    rnn = RNN(parameters['INPUT_SIZE'], parameters["HIDDEN_SIZE"], parameters['NUM_LAYERS'], parameters['OUTPUT_SIZE'])
    optimizer = torch.optim.Adam(rnn.parameters(), lr=parameters['learning_rate'])
    criterion = nn.MSELoss()
    hidden_state = None

    for epoch in range(parameters['num_epochs']):
        inputs = Variable(torch.from_numpy(x_train).float())
        labels = Variable(torch.from_numpy(y_train).float())
        output, hidden_state = rnn(inputs, hidden_state)
        loss = criterion(output.view(-1), labels)
        optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        loss.backward(retain_graph=True)
        optimizer.step()

        print('epoch{}, loss{}'.format(epoch, loss.item()))
    
    return rnn
