import torch.nn as nn


time_step = 128  # time step/image height
input_size = 12  # input_size / image width


class Rnn(nn.Module):
    def __init__(self):
        super(Rnn, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        # 定义回归层网络，输入的特征维度等于LSTM的输出，输出维度为1
        self.out = nn.Linear(128, 4)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # x (batch,time_step,input_size)
        out = self.out(r_out[:, -1, :])  # (batch,time_step,input)
        return out





