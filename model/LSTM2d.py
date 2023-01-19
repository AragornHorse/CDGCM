from . import blocks
# import blocks
import torch
import torch.nn as nn
import torch.optim as optim


class _LSTM2d(nn.Module):
    def __init__(self, in_channel, cls_size, in_size, bias, dropout, num_layer, device):
        super(_LSTM2d, self).__init__()
        self.lstm = [blocks.LSTM2d(in_channel, in_channel*2, out_channel=in_channel*2, memory_channel=in_channel*2,
                                   device=device, bias=False, dropout=dropout, stride=2).to(device)]
        num = (in_size + 1) // 2
        channel = in_channel * 2

        for _ in range(num_layer-1):
            self.lstm.append(
                blocks.LSTM2d(channel, channel*2, out_channel=channel*2, memory_channel=channel*2,
                              device=device, bias=False, dropout=dropout, stride=2).to(device)
            )
            channel *= 2
            num = (num + 1) // 2

        self.fc = nn.Sequential(
            nn.Linear(channel * num ** 2, cls_size),
        )

    def forward(self, x, memorys=None):   # b, s, c, w, h
        b, s, c, w, h = x.size()
        memorys = memorys
        out = x[:, 0]
        if memorys is None:
            memorys = []
            for idx, layer in enumerate(self.lstm):
                out, memory = layer(out)
                memorys.append(memory)
        else:
            for idx, layer in enumerate(self.lstm):
                out, memorys[idx] = layer(out, memorys[idx])
        for i in range(1, s):
            out = x[:, i]
            for idx, layer in enumerate(self.lstm):
                out, memorys[idx] = layer(out, memorys[idx])
        out = self.fc(out.reshape(b, -1))
        return out, memorys


class LSTM2d:
    def __init__(self, in_channel=1, cls_size=29, in_size=100, bias=False, dropout=0.4, num_layer=4,
                 opt=optim.Adam, lr=1e-3, device=torch.device("cpu")):
        self.model = _LSTM2d(in_channel, cls_size, in_size, bias, dropout, num_layer, device).to(device)

        self.loss_func = nn.CrossEntropyLoss()

        self.device = device

        self.opt = opt(self.model.parameters(), lr=lr)

    def train(self, data):
        self.model.train()
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        out, _ = self.model(x)

        loss = self.loss_func(out, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        cor = torch.sum(torch.argmax(out, -1) == y) / y.size(0)

        return loss, cor

    def save(self, path=r"./parameters/lstm3d.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path=r"./parameters/lstm3dm.pth"):
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
