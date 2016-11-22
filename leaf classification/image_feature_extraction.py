import numpy as np
import os, sys
import matplotlib.pyplot as plt


__author__ = 'Ming Li'
# This application forms a submission from Ming in regards to leaf classification challenge on Kaggle community

path = 'leaf/images/'
pic_names = {i: path + str(i) + '.jpg' for i in range(1, 1585, 1)}

# exploring possible feature

pic = pic_names[99]

print(pic)

# img = mpimg.imread(pic)
# cy, cx = ndi.center_of_mass(img)
#
# # extract shape of the image
#
# contours = measure.find_contours(array=img, level=.8)
#
# # find largest contour among all possibilities
#
# contour = max(contours, key=len)
#
# # demean contour coordinates
#
# polar_contour = contour.copy()
# polar_contour[:, 0] -= cy
# polar_contour[:, 1] -= cx
#
#
# # let us see the contour that we hopefully found
# plt.plot(polar_contour[::, 1], polar_contour[::, 0], linewidth=1, color='black')  # (I will explain this [::,x] later)
# plt.scatter(0, 0)
# plt.title(pic)
# plt.clf()
#
# # for local maxima
# c_max_index = argrelextrema(polar_contour[:, 0], np.greater, order=50)
# c_min_index = argrelextrema(polar_contour[:, 0], np.less, order=50)
#
# plt.scatter(polar_contour[:, 1], polar_contour[:, 0],
#             linewidth=0, s=2, c='k')
# plt.scatter(polar_contour[:, 1][c_max_index],
#             polar_contour[:, 0][c_max_index],
#             linewidth=0, s=30, c='b')
# plt.scatter(polar_contour[:, 1][c_min_index],
#             polar_contour[:, 0][c_min_index],
#             linewidth=0, s=30, c='r')
#
#
# plt.show()
