from . import blocks
import torch
import torch.nn as nn
import torch.optim as optim


class _Resnet(nn.Module):
    def __init__(self, in_channel, cls_num=29, res_num=6, in_size=100, dropout=0.4):
        super(_Resnet, self).__init__()
        channel = in_channel
        lst = [
            blocks.ResBlock(channel, channel*2, stride=1, dropout=dropout)
        ]
        channel *= 2
        num = in_size
        for _ in range(res_num):
            lst.append(
                blocks.ResBlock(channel, 2 * channel, stride=2, shortcut_size=3, conv_size=3, padding=1,
                                bias=False, dropout=dropout)
            )
            channel *= 2
            num = (num + 1) // 2
        self.res_layers = nn.Sequential(*lst)
        self.fc = nn.Sequential(
            nn.Linear(num**2*channel, cls_num)
        )

    def forward(self, x):
        x = x.squeeze()
        # x = x + torch.linspace(-1, 1, steps=x.size(1), device=x.device).reshape(1, -1, 1, 1)
        out = self.res_layers(x).reshape(x.size(0), -1)
        out = self.fc(out)
        return out


class ResNet:
    def __init__(self, in_channel, cls_num=29, res_num=6, in_size=100, dropout=0.4, lr=1e-3,
                 device=torch.device("cpu"), opt=optim.Adam):

        self.model = _Resnet(in_channel, cls_num=cls_num, res_num=res_num, in_size=in_size, dropout=dropout).to(device)

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

    def save(self, path=r"./parameters/resnet.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path=r"./parameters/resnet.pth"):
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




