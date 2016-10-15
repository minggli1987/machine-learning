import numpy as np
from sklearn.feature_extraction import image, img_to_graph
import os, sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import measure
import scipy.ndimage as ndi

__author__ = 'Ming Li'
# This application forms a submission from Ming in regards to leaf classification challenge on Kaggle community

path = 'leaf/images/'
pic_names = {i: path + str(i) + '.jpg' for i in range(1, 1585, 1)}

pic = pic_names[20]

print(pic)

img = mpimg.imread(pic)

cy, cx = ndi.center_of_mass(img)

plt.imshow(img, cmap='Set3')
plt.title(pic)
# plt.scatter(cx, cy)
# # plt.show()

# extract shape of the image

contours = measure.find_contours(array=img, level=.8)

# find largest contour among all possibilities

contour = max(contours, key=len)

# demean contour coordinates

contour[:, 1] -= cx
contour[:, 0] -= cy

# let us see the contour that we hopefully found
plt.plot(contour[:, 1], contour[:, 0], linewidth=0.5)  # (I will explain this [::,x] later)
plt.show()

