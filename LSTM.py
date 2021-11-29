import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.input_size = 4  # 4 bands
        self.hidden_size = 64
        self.proj_size = 52  # 52 kinds of labels

        self.codename = 'LSTM'
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            proj_size=self.proj_size,
            num_layers=1,
            batch_first=True,
        )
        # self.out = nn.Linear(64, 10)
        # self.out = nn.Linear(self.hidden_size, self.proj_size)

    def forward(self, x):
        r_out, _ = self.lstm(x)  # r_out contains the output features (h_t) from the last layer of the LSTM
        result = r_out[:, -1, :]
        return result

