import torch
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import numpy as np
import random


def path2grayArray(path, size=None):
    if size is None:
        size = [100, 100]
    img = Image.open(path).resize(size).convert('L')
    return np.array(img)


class _DataSet(Dataset):
    def __init__(self, index_path=r"C:\Users\DELL\Desktop\datasets\jester\index", size=None):
        self.Idx2PathNum_Path = index_path + r"\idx2path&num.txt"
        with open(self.Idx2PathNum_Path, 'r') as f:
            self.lst = [[line[:-1].split(';')[1], int(line[:-1].split(';')[2])] for line in f]
        if size is None:
            size = [100, 100]
        self.size = size

    def __getitem__(self, item):
        path, num = self.lst[item]
        video = [path2grayArray(file) for file in glob.glob(path + r"\*")]
        return np.array(video)

    def __len__(self):
        return len(self.lst)


class DataSet(Dataset):
    def __init__(self, index_path=r"C:\Users\DELL\Desktop\datasets\jester\index", size=None):
        self.data = _DataSet(index_path, size)
        self.length = len(self.data)

    def __getitem__(self, item):
        num = random.randint(1, 10)
        videos = []
        labels = []

        for _ in range(num):
            idx = random.randint(0, self.length-1)
            video = self.data[idx]   # s, c, w, h
            label = np.zeros(video.shape[0])  # s
            label[-6:-1] = label[-6:-1] + 1
            videos.append(video)
            labels.append(label)

        videos = np.concatenate(videos, axis=0)
        labels = np.concatenate(labels, axis=0)

        return torch.tensor(videos, dtype=torch.float).unsqueeze(0).unsqueeze(2),\
               torch.tensor(labels, dtype=torch.float).unsqueeze(0)

    def __len__(self):
        return self.length















