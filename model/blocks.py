import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, shortcut_size=3, act_func=nn.LeakyReLU(inplace=True),
                 conv_size=3, padding=1, bias=False, dropout=0.4):
        super(ResBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channel, out_channel, kernel_size=conv_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(out_channel),
        )
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=shortcut_size, stride=stride, padding=padding,
                          bias=bias),
                nn.BatchNorm2d(out_channel),
            )
        self.act_func = act_func

    def forward(self, x):
        layer_out = self.layer(x)
        shortcut = self.shortcut(x)
        out = self.act_func(layer_out + shortcut)
        return out


class Attention(nn.Module):
    def __init__(self, in_size, out_size, qk_size=16, v_bias=True, qk_bias=True):
        super(Attention, self).__init__()

        self.v = nn.Linear(in_size, out_size, bias=v_bias)
        self.q = nn.Linear(in_size, qk_size, bias=qk_bias)
        self.k = nn.Linear(in_size, qk_size, bias=qk_bias)

    def forward(self, x):
        v = self.v(x)
        q = self.q(x)
        k = self.k(x)

        attn = torch.softmax(q @ torch.transpose(k, dim1=-1, dim0=-2), dim=-1)
        out = attn @ v

        return out


class Conv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True,
                 act_func=nn.LeakyReLU(inplace=True)):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.act_func = act_func

    def forward(self, x):
        x = torch.transpose(x, dim0=1, dim1=2)
        out = self.conv(x)
        out = self.act_func(out)
        return torch.transpose(out, dim0=1, dim1=2)


class Res3d(nn.Module):
    def __init__(self, in_channel, out_channel, stride, padding, kernel_size=3, bias=False, dropout=0.4):
        super(Res3d, self).__init__()
        self.long = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=bias),
            nn.BatchNorm3d(out_channel),
            nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm3d(out_channel)
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride,
                      bias=bias),
            nn.BatchNorm3d(out_channel)
        )
        self.act_func = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out1 = self.long(x)
        out2 = self.shortcut(x)
        out = self.act_func(out1 + out2)
        return out


class LSTM2d(nn.Module):
    def __init__(self, in_channel, h_channel, out_channel, memory_channel, device, bias=True, dropout=0.4, stride=1):
        super(LSTM2d, self).__init__()
        self.h = nn.Conv2d(in_channel + h_channel, h_channel, bias=True, stride=1, padding=1, kernel_size=3)
        self.c = nn.Conv2d(h_channel, memory_channel, bias=bias, stride=1, kernel_size=3, padding=1)

        self.f = nn.Sequential(
            nn.Conv2d(in_channel + h_channel, 1, bias=bias, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.memory_channel = memory_channel
        self.h_channel = h_channel

        self.i = nn.Sequential(
            nn.Conv2d(in_channel + h_channel, 1, bias=bias, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.device = device

        self.out = ResBlock(memory_channel, out_channel, stride, padding=1, bias=bias, dropout=dropout)

    def forward(self, x, memory=None):  # (c0, h0)
        if memory is None or len(memory) != 2:
            c0 = torch.zeros(x.size(0), self.memory_channel, x.size(-2), x.size(-1)).to(self.device)
            h0 = torch.zeros(x.size(0), self.h_channel, x.size(-2), x.size(-1)).to(self.device)
        else:
            c0, h0 = memory

        input = torch.concat([x, h0], dim=1)
        h = self.h(input)
        f = self.f(input)
        i = self.i(input)
        c = self.c(h)
        c = f * c0 + i * c
        out = self.out(c)
        return out, (c, h)


