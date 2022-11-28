import torch.nn as nn

in_dim = 12
hidden_dim = 128
n_layer = 2
n_class = 4


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_class)

    def forward(self, x):  # x‘s shape (batch_size, 序列长度, 序列中每个数据的长度)
        out, _ = self.lstm(x)  # out‘s shape (batch_size, 序列长度, hidden_dim)
        out = out[:, -1, :]  # 中间的序列长度取-1，表示取序列中的最后一个数据，这个数据长度为hidden_dim，
        # 得到的out的shape为(batch_size, hidden_dim)
        out = self.linear(out)  # 经过线性层后，out的shape为(batch_size, n_class)
        print(out.shape)
        return out
