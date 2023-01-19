from torch import fft
from gesture.manage_data.dataloader import DataSet
import matplotlib.pyplot as plt
import numpy as np

# img = DataSet()[4000][0][10].squeeze().numpy()
#
# # plt.imshow(img)
# # plt.show()
#
# f = np.fft.fft2(img)
#
# max = np.max(np.absolute(f))
# mean = np.mean(np.absolute(f))
# min = np.min(np.absolute(f))
# down = 0.999 * mean + 0.001 * max + 0.0 * min
# top = 0.001 * mean + 0.999 * max + 0 * min
#
# img = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.where(np.absolute(f) > top, 0, f))))
# print(img)
#
# plt.imshow(np.abs(img))
# plt.show()
#
# fshift = 20*np.log(np.abs(np.fft.fftshift(f)))


def filter(img):
    img = np.array(img)
    f = np.fft.fft2(img)
    ab = np.absolute(f)
    max = np.max(ab)
    mean = np.mean(ab)
    min = np.min(ab)
    down = 0.999 * mean + 0.001 * max + 0.0 * min
    top = 0.001 * mean + 0.999 * max + 0 * min
    img = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.where(ab > top, 0, f))))
    return np.abs(img)

# plt.imshow(fshift)
# plt.show()







