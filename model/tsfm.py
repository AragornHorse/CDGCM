# import blocks
import torch
import torch.nn as nn
import torch.optim as optim
from . import blocks


class _GT(nn.Module):
    def __init__(self, in_channel, res_num=2, head_channel=4, dropout=0.4, bias=False, in_size=100, tsfm_layers=4,
                 n_head=8, dim_forward=512, cls_size=29):
        super(_GT, self).__init__()
        self.head = nn.Sequential(
            blocks.ResBlock(in_channel, head_channel, stride=1, padding=1, bias=bias, dropout=dropout)
        )

        lst = []
        channel = head_channel
        num = in_size
        for _ in range(res_num):
            lst.append(
                blocks.ResBlock(channel, channel*2, stride=2, padding=1, bias=bias, dropout=dropout)
            )
            channel *= 2
            num = (num + 1) // 2
        self.res_layer = nn.Sequential(*lst)

        self.o_channel = channel
        self.o_num = num

        self.convs = nn.Sequential(
            self.head,
            self.res_layer
        )

        self.bert = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=channel * num ** 2, nhead=n_head, dim_feedforward=dim_forward, dropout=dropout,
                batch_first=True, norm_first=True
            ),
            num_layers=tsfm_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(channel * num ** 2, cls_size, bias=bias)
        )

    def forward(self, x):   # b, s, c, w, h
        b, s, c, w, h = x.size()

        out = torch.Tensor(b, s, self.o_channel, self.o_num, self.o_num).to(x.device)
        for i in range(s):
            out[:, i, :, :, :] = self.convs(x[:, i, :, :, :])

        out = out.reshape(b, s, -1)   # b, s, -1
        out = self.bert(out)  # b, s, -1
        out = torch.mean(out, dim=1).squeeze()  # b, -1
        out = self.fc(out)
        return out


class GT:
    def __init__(self,  in_channel=1, res_num=2, head_channel=4, dropout=0.4, bias=False, in_size=100, tsfm_layers=4,
                 n_head=8, dim_forward=512, cls_size=29, lr=1e-3, device=torch.device("cpu"), opt=optim.Adam):
        self.model = _GT(in_channel, res_num, head_channel, dropout, bias,
                         in_size, tsfm_layers, n_head, dim_forward, cls_size).to(device)

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

    def save(self, path=r"./parameters/tsfm.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path=r"./parameters/tsfm.pth"):
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