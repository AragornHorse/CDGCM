from . import blocks
import torch
import torch.nn as nn
import torch.optim as optim


class _Res_Attention(nn.Module):
    def __init__(self, in_channel=1, head_channel=8, head_bias=True, res_num=3, conv3d_num=2, in_size=100, h_attn=128,
                 cls_num=29, dropout=0.4):
        super(_Res_Attention, self).__init__()
        # in: b, s, 1, 100, 100
        self.head = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, head_channel, kernel_size=3, stride=1, padding=1, bias=head_bias),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(head_channel, head_channel, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(head_channel)
        )
        # b, s, head, 100, 100
        num = in_size
        lst = []
        channel = head_channel
        for _ in range(res_num):
            lst.append(
                blocks.ResBlock(channel, 2 * channel, stride=2, shortcut_size=3, conv_size=3, padding=1,
                                bias=False, dropout=dropout)
            )
            channel *= 2
            num = (num + 1)//2
        self.res_layers = nn.Sequential(*lst)
        # b, s, channel, 100/2^res, 100/2^res

        lst = []
        for _ in range(conv3d_num):
            lst.append(
                blocks.Conv3d(channel, 2 * channel, kernel_size=3, stride=2, padding=1, bias=False)
            )
            channel *= 2
            num = (num + 1)//2
        self.conv3d_layers = nn.Sequential(*lst)
        # b, s//2^conv, channel, size, size

        self.attn_layers = nn.Sequential(
            blocks.Attention(channel * num ** 2, h_attn, qk_size=16, v_bias=False, qk_bias=False),
            nn.LayerNorm(h_attn),
            blocks.Attention(h_attn, h_attn, qk_size=16, v_bias=False, qk_bias=False),
            nn.LeakyReLU(inplace=True),
        )
        # b, s//2^conv, h_attn

        self.fc = nn.Sequential(
            nn.LayerNorm(h_attn),
            nn.Linear(h_attn, cls_num, bias=False),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(-1, x.size(2), x.size(3), x.size(4))
        out = self.head(x)
        out = self.res_layers(out)
        out = out.reshape(batch_size, -1, out.size(-3), out.size(-2), out.size(-1))
        out = self.conv3d_layers(out)
        out = out.reshape(batch_size, out.size(1), -1)
        out = self.attn_layers(out)
        out = self.fc(out)
        out, _ = torch.max(out, dim=-2)
        return out

# x = torch.ones(16, 30, 1, 100, 100)
# print(Res_Attention_()(x).size())

class Res_Attention:
    def __init__(self, in_channel=1, head_channel=8, head_bias=True, res_num=3, conv3d_num=2, in_size=100, h_attn=128,
                 cls_num=29, dropout=0.4, lr=1e-3, device=torch.device("cpu"), opt=optim.Adam):

        self.model = _Res_Attention(in_channel, head_channel, head_bias, res_num, conv3d_num,
                                    in_size, h_attn, cls_num, dropout).to(device)

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

    def save(self, path=r"./parameters/res_attn.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path=r"./parameters/res_attn.pth"):
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





