from gesture.manage_data.dataloader import DataSet
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np


img = Image.open(r"C:\Users\DELL\Desktop\datasets\jester\jester1\4000\00020.jpg")

plt.imshow(img)
plt.show()

img = img.filter(ImageFilter.BLUR)
plt.imshow(img)
plt.show()


img = img.filter(ImageFilter.CONTOUR)

plt.imshow(img)
plt.show()

img = img.convert('L')

plt.imshow(img)
plt.show()


# img = DataSet(fft=False, contour=True, find_edges=True, filter=ImageFilter.SMOOTH)[110][0][12].squeeze().numpy()
#
# plt.imshow(img)
# plt.show()










