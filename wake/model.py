import torch
import torch.nn as nn
import torch.optim as optim
from gesture.model import blocks


class _WL(nn.Module):
    def __init__(self, in_size=100, in_channel=1, res_num=4, lstm_n=2, bias=False, dropout=0.4, lstm_dim=128):
        super(_WL, self).__init__()
        lst = []
        channel = in_channel
        num = in_size

        for _ in range(res_num):
            lst.append(
                blocks.ResBlock(channel, channel * 2, stride=2, padding=1, bias=bias, dropout=dropout)
            )
            channel *= 2
            num = (num + 1) // 2
        self.res_layer = nn.Sequential(*lst)

        self.lstm = nn.LSTM(input_size=channel * num ** 2, hidden_size=lstm_dim, bidirectional=False,
                            batch_first=True, num_layers=lstm_n, bias=bias, dropout=dropout)

        self.fc = nn.Sequential(
            nn.Linear(lstm_dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, cell=None):
        b, s, c, w, h = x.size()
        out = self.res_layer(x.reshape(-1, c, w, h))
        _, c, w, h = out.size()
        if cell is None:
            out, c = self.lstm(out.reshape(b, s, -1))
        else:
            out, c = self.lstm(out.reshape(b, s, -1), cell)
        out = self.fc(out).squeeze()  # b, s
        return out, c


class WL:
    def __init__(self, in_size=100, in_channel=1, res_num=4, lstm_n=2, bias=False, dropout=0.4, lstm_dim=128,
                 lr=1e-3, device=torch.device("cpu")):
        self.model = _WL(in_size, in_channel, res_num, lstm_n, bias, dropout, lstm_dim).to(device)

        self.loss_func = nn.BCELoss()

        self.opt = optim.Adam(self.model.parameters(), lr=lr)

        self.device = device

        self.cell = None

    def train(self, data, cell=None):
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)
        out, cell = self.model(x, self.cell)  # 1, s
        a, b = cell
        self.cell = (a.detach(), b.detach())
        arg_out = (out > 0.5) + 0  # 1, s

        loss = 0.15 * self.loss_func((arg_out.detach() == 1) * out, (arg_out.detach() == 1) * y.squeeze()) + \
               0.8 * self.loss_func((y == 1) * out, (y == 1) * y.squeeze()) +   \
               0.05 * self.loss_func(out, y.squeeze())  # pos的要准->减小运算; true的要准->不漏掉手势; 总体要准

        # loss = self.loss_func(out, y.squeeze())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        arg_out = (out > 0.5) + 0  # 1, s
        cor = torch.sum(arg_out == y) / y.size(1)

        cor1 = torch.sum((arg_out == 1) * (y == 1)) / (torch.sum((arg_out == 1) + 0) + 1e-6)  # 预测是1的有几个真是1
        cor2 = torch.sum((arg_out == 1) * (y == 1)) / (torch.sum((y == 1) + 0) + 1e-6)  # 是1的有几个被预测出来了

        return loss, cor, cor1, cor2, (arg_out.squeeze().detach(), y.squeeze())
