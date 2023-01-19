import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import glob


class DataSet(Dataset):
    def __init__(self, size=None, mode='train'):
        if size is None:
            size = [50, 50]
        self.size = size
        # self.lst_face = glob.glob(r"C:\Users\DELL\Desktop\datasets\jester\jester1\*\*")
        # self.lst_no_face = glob.glob(r"C:\Users\DELL\Desktop\datasets\jester\no_face\*")
        with open(r"C:\Users\DELL\Desktop\datasets\jester\no_face.txt", 'r') as f:
            self.lst_no_face = [line[:-1] for line in f]
        with open(r"C:\Users\DELL\Desktop\datasets\jester\have_face.txt", 'r') as f:
            self.lst_face = [line[:-1] for line in f]
        # with open(r"C:\Users\DELL\Desktop\datasets\jester\no_face.txt", 'w') as f:
        #     for line in self.lst_no_face:
        #         f.write(line)
        #         f.write("\n")
        # with open(r"C:\Users\DELL\Desktop\datasets\jester\have_face.txt", 'w') as f:
        #     for line in self.lst_face:
        #         f.write(line)
        #         f.write("\n")

    def __getitem__(self, item):
        if item % 2 == 1:
            path = self.lst_face[(item // 2) % len(self.lst_face)]
            label = 1
        else:
            path = self.lst_no_face[(item // 2) % len(self.lst_no_face)]
            label = 0
        img = Image.open(path).resize(self.size).convert('L')
        return torch.tensor(np.array(img), dtype=torch.float).unsqueeze(0), label

    def __len__(self):
        return 2 * max(len(self.lst_face), len(self.lst_no_face))


# import matplotlib.pyplot as plt
#
# d = DataSet()
#
# for i in d:
#     img, label = i
#     plt.imshow(img)
#     plt.show()
#     print(img.shape)
#     input()

def load(mode='train', batch_size=64, size=None):
    return DataLoader(DataSet(size=size, mode=mode), batch_size=batch_size)

