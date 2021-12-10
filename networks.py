import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.codename = 'RNN'
        # -> x needs to be: (batch_size, seq, input_size)
        
        # or:
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  
        #out, _ = self.lstm(x, (h0,c0))  
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional = False):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.codename = 'LSTM'
        self.bid = bidirectional
        # -> x needs to be: (batch_size, seq, input_size)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        if self.bid == True:
            self.fc = nn.Linear(2*hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        if self.bid == True:
            h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        # out, _ = self.rnn(x, h0)  
        out, _ = self.lstm(x, (h0,c0))  
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, output_size, bidirectional = False):
        super(GRU, self).__init__()
        self.codename = 'GRU'
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_size)
        self.bid = bidirectional
        
        if self.bid == True:
            self.fc = nn.Linear(2*hidden_dim, output_size)
            

    def forward(self, X):
        out,hidden = self.gru(X)
        out = out[:, -1, :]
        out = self.fc(out)
        return out