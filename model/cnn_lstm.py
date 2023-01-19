from . import blocks
# import blocks
import torch
import torch.nn as nn
import torch.optim as optim


class _cnn_lstm(nn.Module):
    def __init__(self, in_size, in_channel, res_num, bias, dropout, lstm_num, res_outsize, lstm_hsize, cls_size):
        super(_cnn_lstm, self).__init__()
        lst = [
            blocks.ResBlock(in_channel=in_channel, out_channel=in_channel*2, stride=1, padding=1, dropout=dropout)
        ]
        channel = in_channel * 2
        num = in_size
        for _ in range(res_num):
            lst.append(
                blocks.ResBlock(channel, channel*2, stride=2, padding=1, bias=bias, dropout=dropout)
            )
            channel *= 2
            num = (num + 1) // 2
        self.resnet = nn.Sequential(*lst)
        self.res_out = nn.Sequential(
            nn.Linear(channel * num ** 2, res_outsize)
        )

        self.rnn = nn.LSTM(input_size=res_outsize, hidden_size=lstm_hsize, bidirectional=False, batch_first=True,
                           num_layers=lstm_num, dropout=dropout)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(lstm_hsize),
            nn.Linear(lstm_hsize, cls_size)
        )

    def forward(self, x):   # b, s, c, w, h
        b, s, c, w, h = x.size()
        memory = None
        for i in range(s):
            u = x[:, i]   # b, c, w, h
            out = self.resnet(u).reshape(b, -1)
            out = self.res_out(out)  # b, -1
            out, memory = self.rnn(out, memory) if memory is not None else self.rnn(out)
        out = self.fc(out)
        return out


class CNN_LSTM:
    def __init__(self, in_size=100, in_channel=1, res_num=3, bias=False, dropout=0.4, lstm_num=2, res_outsize=128,
                 lstm_hsize=128, cls_size=29, device=torch.device("cpu"), opt=optim.Adam, lr=1e-3):
        self.model = _cnn_lstm(in_size, in_channel, res_num, bias,
                               dropout, lstm_num, res_outsize, lstm_hsize, cls_size).to(device)

        self.loss_func = nn.CrossEntropyLoss()

        self.device = device

        self.opt = opt(self.model.parameters(), lr=lr)

    def train(self, data):
        self.model.train()
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        out = self.model(x)

        loss = self.loss_func(out, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        cor = torch.sum(torch.argmax(out, -1) == y) / y.size(0)

        return loss, cor

    def save(self, path=r"./parameters/cnn_lstm.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path=r"./parameters/cnn_lstm.pth"):
        self.model.load_state_dict(torch.load(path))

    def eval(self, data):
        self.model.eval()
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        out = self.model(x)

        loss = self.loss_func(out, y)

        cor = torch.sum(torch.argmax(out, -1) == y) / y.size(0)

        return loss, cor

