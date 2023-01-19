from copy import copy
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import random


def fit_video_len(video: list, tgt_len=30):
    if len(video) >= tgt_len:
        return video
    idx = random.randint(0, len(video) - 1)
    video.insert(idx, copy(video[idx]))
    return fit_video_len(video, tgt_len)


def path2grayArray(path, size=None, fft=True, contour=False, find_edges=True, filter=None):
    if size is None:
        size = [100, 100]
    img = Image.open(path).resize(size)

    if filter is not None:
        img = img.filter(filter)

    if contour:
        img = img.filter(ImageFilter.CONTOUR)
    elif find_edges:
        img = img.filter(ImageFilter.FIND_EDGES)

    img = img.convert('L')
    if fft:
        from gesture.fft.fft import filter
        img = filter(img)
    else:
        img = np.array(img)
    return img


class DataSet(Dataset):
    def __init__(self, index_path=r"C:\Users\DELL\Desktop\datasets\jester\index", size=None,
                 tgt_len=30, fft=True, contour=False, find_edges=True, filter=None, mode='train'):
        if mode == 'train':
            self.Idx2PathNum_Path = index_path + r"\idx2path&num.txt"
        else:
            self.Idx2PathNum_Path = index_path + r"\idx2path&num_validation.txt"

        with open(self.Idx2PathNum_Path, 'r') as f:
            self.lst = [[line[:-1].split(';')[1], int(line[:-1].split(';')[2])] for line in f]
        if size is None:
            size = [100, 100]
        self.size = size
        self.tgt_len = tgt_len
        self.fft = fft
        self.contour = contour
        self.find_edges = find_edges
        self.filter = filter

    def __getitem__(self, item):
        path, num = self.lst[item]
        video = [
            path2grayArray(file, fft=self.fft, contour=self.contour, find_edges=self.find_edges, filter=self.filter)
            for file in glob.glob(path + r"\*")
        ]
        video = fit_video_len(video, tgt_len=self.tgt_len)
        return torch.tensor(np.array(video[:self.tgt_len]), dtype=torch.float).unsqueeze(1), num

    def __len__(self):
        return len(self.lst)


def load(batch_size=32, size=None, tgt_len=30, fft=True, mode='train', contour=False, find_edges=True, filter=None):
    if size is None:
        size = [100, 100]
    return DataLoader(
        DataSet(size=size, tgt_len=tgt_len, fft=fft, mode=mode, contour=contour, find_edges=find_edges, filter=filter)
        , batch_size=batch_size, shuffle=True
    )

# for d in DataSet():
#     print(d[0].size())
