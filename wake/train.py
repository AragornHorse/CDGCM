from dataset import DataSet
from model import WL
import torch

model = WL(in_size=100, in_channel=1, res_num=4, lstm_n=2, lstm_dim=128, bias=False, lr=1e-3, dropout=0.4,
           device=torch.device("cuda"))
dataset = DataSet()

num = 0

for epoch in range(1):
    for data in dataset:
        num += 1
        loss, cor, cor1, cor2, rst = model.train(data)
        print("loss:{}, cor:{}, true/pos:{}, pos/true:{}".format(loss, cor, cor1, cor2))
        if num > 100:
            out, y = rst
            print("out>>>")
            print(out)
            print("label>>>")
            print(y)
            num = 0




