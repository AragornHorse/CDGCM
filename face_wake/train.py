import torch
from dataset import load
from model import Res


model = Res(in_size=50, in_channel=1, layer_num=6, dropout=0.4, device=torch.device("cuda"), lr=1e-3)

loader = load("train", batch_size=64, size=[50, 50])

for epoch in range(1):
    for data in loader:
        loss, cor = model.train(data)
        print("loss:{}, cor:{}".format(loss, cor))


