from . import blocks
import torch
import torch.nn as nn
import torch.optim as optim
# import blocks


class _pure3d(nn.Module):
    def __init__(self, in_channel=1, in_size=100, dropout=0.4, head_channel=4, res_num=6, seq=30,
                 cls_size=29):
        super(_pure3d, self).__init__()
        self.head = nn.Sequential(
            blocks.Res3d(in_channel, head_channel, stride=1, padding=1, bias=False, dropout=dropout)
        )
        channel = head_channel
        num = in_size
        seq = seq
        lst = []
        for _ in range(res_num):
            lst.append(
                blocks.Res3d(channel, channel*2, stride=2, padding=1, bias=False, dropout=dropout)
            )
            channel *= 2
            num = (num + 1) // 2
            seq = (seq + 1) // 2
        self.res_layer = nn.Sequential(*lst)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(channel * seq * num ** 2),
            nn.Linear(channel * seq * num ** 2, cls_size)
        )

    def forward(self, x):   # b, s, c, w, h
        x = torch.transpose(x, dim0=1, dim1=2)  # b, c, s, w, h
        out = self.head(x)
        out = self.res_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class _Res3d(nn.Module):
    def __init__(self, in_channel=1, in_size=100, res2d_num=2, dropout=0.4, head_channel=4, res3d_num=6, seq=30,
                 cls_size=29, tail_channel=4):
        super(_Res3d, self).__init__()
        self.head = nn.Sequential(
            blocks.ResBlock(in_channel, head_channel, stride=1, padding=1, bias=False, dropout=dropout)
        )

        lst = []
        channel = head_channel
        num = in_size
        for _ in range(res2d_num):
            lst.append(
                blocks.ResBlock(channel, channel*2, stride=2, padding=1, dropout=dropout, bias=False)
            )
            channel *= 2
            num = (num + 1) // 2
        self.res2d_layer = nn.Sequential(*lst)

        seq = seq
        lst = []
        for _ in range(res3d_num):
            lst.append(
                blocks.Res3d(channel, 2*channel, stride=2, padding=1, bias=False, dropout=dropout)
            )
            channel *= 2
            num = (num + 1) // 2
            seq = (seq + 1) // 2
        self.res3d_layer = nn.Sequential(*lst)

        self.tail = nn.Sequential(
            nn.Conv2d(seq, tail_channel, bias=False, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(inplace=True)
        )
        # print(tail_channel * num ** 2 * channel)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(tail_channel * num ** 2 * channel),
            nn.Linear(tail_channel * num ** 2 * channel, cls_size)
        )

    def forward(self, x):  # b, s, c, w, h
        b, s, c, w, h = x.size()
        out = self.head(x.reshape(-1, c, w, h))
        out = self.res2d_layer(out)
        _, c, w, h = out.size()
        out = out.reshape(b, s, c, w, h)
        out = torch.transpose(out, dim1=1, dim0=2)  # b, c, s, w, h
        out = self.res3d_layer(out)  # b, c, s, w, h
        b, c, s, w, h = out.size()
        out = out.reshape(-1, s, w, h)
        out = self.tail(out)  # bc, s, w, h
        out = out.reshape(b, -1)  # b, -1
        out = self.fc(out)

        return out

class Res3d:
    def __init__(self, in_channel=1, in_size=100, res2d_num=2, dropout=0.4, head_channel=4, res3d_num=6, seq=30,
                 cls_size=29, tail_channel=4, lr=1e-3, device=torch.device("cpu"), opt=optim.Adam):
        # self.model = _Res3d(in_channel, in_size, res2d_num, dropout, head_channel, res3d_num, seq, cls_size,
        #                     tail_channel).to(device)
        self.model = _pure3d(in_channel, in_size, dropout, head_channel, res3d_num, seq, cls_size).to(device)

        self.loss_func = nn.CrossEntropyLoss()

        self.device = device

        self.opt = opt(self.model.parameters(), lr=lr)

        self.layer = res3d_num
        self.head_channel = head_channel

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

    def save(self, path=None):
        if path is None:
            path = r"./parameters/res3d_h{}_l{}.pth".format(self.head_channel, self.layer)
        torch.save(self.model.state_dict(), path)

    def load(self, path=None):
        if path is None:
            path = r"./parameters/res3d_h{}_l{}.pth".format(self.head_channel, self.layer)
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



