import torch
import torch.nn as nn
import torch.optim as optim
from gesture.model import blocks


class _Res(nn.Module):
    def __init__(self, in_size, in_channel, layer_num, dropout):
        super(_Res, self).__init__()
        lst = []
        channel = in_channel
        size = in_size
        for _ in range(layer_num):
            lst.append(
                blocks.ResBlock(in_channel=channel, out_channel=2*channel, stride=2, padding=1, bias=False,
                                dropout=dropout)
            )
            channel *= 2
            size = (size + 1) // 2
        self.res_layers = nn.Sequential(*lst)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(channel * size ** 2),
            nn.Linear(channel * size ** 2, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, w, h = x.size()
        out = self.res_layers(x).reshape(b, -1)
        out = self.fc(out)
        out = self.sigmoid(out).squeeze()
        return out


class Res:
    def __init__(self, in_size=50, in_channel=1, layer_num=5, dropout=0.4, device=torch.device("cpu"), lr=1e-3):
        self.model = _Res(in_size, in_channel, layer_num, dropout).to(device)

        self.loss_func = nn.BCELoss()

        self.device = device

        self.opt = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, data):
        self.model.train()
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device).float()

        out = self.model(x)

        loss = self.loss_func(out, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        cor = torch.sum(((out > 0.5) == y) + 0) / y.size(0)

        return loss, cor

    def save(self, path=r"./Res.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path=r"./Res.pth"):
        self.model.load_state_dict(torch.load(path))

    def eval(self, data):
        self.model.eval()
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device).float()

        out = self.model(x)

        loss = self.loss_func(out, y)

        cor = torch.sum(((out > 0.5) == y) + 0) / y.size(0)

        return loss, cor





