import torch
from manage_data.dataloader import load
from model.LSTM2d import LSTM2d
from model.res_attn import Res_Attention
from model.resnet import ResNet
from model.res3d import Res3d
from model.tsfm import GT
from model.cnn_lstm import CNN_LSTM
import torch.optim as optim
from PIL import ImageFilter

# model = Res_Attention(in_channel=1, in_size=100, head_channel=8, head_bias=True, res_num=4, conv3d_num=3,
#                       opt=optim.Adam, h_attn=128, cls_num=29, dropout=0.2, lr=1e-2, device=torch.device("cuda"))
# model = ResNet(in_channel=30, cls_num=29, res_num=6, in_size=100, dropout=0.4, lr=1e-5,
#                device=torch.device("cuda"), opt=optim.Adam)
model = Res3d(in_channel=1, in_size=100, res2d_num=0, dropout=0.8, head_channel=4, res3d_num=6, seq=30, cls_size=29,
              tail_channel=2, lr=3e-6, device=torch.device("cuda"), opt=optim.Adam)
# model = GT(in_channel=1, in_size=100, res_num=4, dropout=0, bias=False, head_channel=2, tsfm_layers=2, n_head=4,
#            dim_forward=256, cls_size=29, lr=1e-3, device=torch.device("cuda"), opt=optim.Adam)
# model = CNN_LSTM(in_channel=1, in_size=100, res_num=3, bias=False, dropout=0.4, lstm_num=2, res_outsize=128,
#                  lstm_hsize=128, cls_size=29, device=torch.device("cuda"), opt=optim.Adam, lr=1e-3)
# model = LSTM2d(in_channel=1, cls_size=29, in_size=100, bias=False, dropout=0.4, num_layer=4,
#                opt=optim.Adam, lr=1e-3, device=torch.device("cuda"))


loader = load(batch_size=64, fft=False, contour=True, find_edges=False, filter=ImageFilter.SMOOTH)
try:
    model.load()
except:
    print("failed to load, maybe you have changed the model-shape, perhaps you should change your saving path")
    pass


for epoch in range(1):
    # model = Res3d(in_channel=1, in_size=100, res2d_num=0, dropout=0.8, head_channel=4, res3d_num=6, seq=30, cls_size=29,
    #               tail_channel=2, lr=lst[epoch], device=torch.device("cuda"), opt=optim.Adam)
    # model.load()
    for data in loader:
        # import matplotlib.pyplot as plt
        # img = data[0][0][15].squeeze().numpy()
        # plt.imshow(img)
        # plt.show()
        loss, cor = model.train(data)
        print("loss:{}, cor:{}".format(loss, cor))
model.save()


model.load()
loader = load(batch_size=64, fft=False, contour=True, find_edges=False, filter=ImageFilter.SMOOTH, mode='eval')
print('\n\n\n\neval>>>')
losses = []
cors= []
for data in loader:
    loss, cor = model.eval(data)
    print("loss:{}, cor:{}".format(loss, cor))
    losses.append(loss.cpu().detach().numpy())
    cors.append(cor.cpu().detach().numpy())

import numpy as np
print("average_loss:{}".format(np.mean(np.array(losses))))
print("average_cor:{}".format(np.mean(np.array(cors))))




