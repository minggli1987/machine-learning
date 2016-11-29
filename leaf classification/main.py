import os
import pandas as pd
import numpy as np
import warnings
import sys
from utils import delete_folders, extract, pic_resize
from GradientDescent import GradientDescent
from sklearn import metrics, model_selection, preprocessing
warnings.filterwarnings('ignore')

# coding: utf-8

__author__ = 'Ming Li'

"""This application forms a submission from Ming Li in regards to leaf classification challenge on Kaggle community"""

dir_path = 'leaf/images/'
label_map, classes = extract('leaf/train.csv')
pic_names = [i.name for i in os.scandir(dir_path) if i.is_file()]

# cross validation of training photos

kf_iterator = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

train_x = list(label_map.keys())  # leaf id
train_y = list(label_map.values())  # leaf species names
leaf_images = dict()  # dictionary of resized leaf images

cross_val = False

for train_index, valid_index in kf_iterator.split(train_x, train_y):

    train_id = [train_x[i] for i in train_index]
    valid_id = [train_x[i] for i in valid_index]

    for _, name in enumerate(pic_names):

        leaf_id = int(name.split('.')[0])
        leaf_images[leaf_id] = pic_resize(dir_path + name)

        if leaf_id in train_id:
            directory = dir_path + 'train/' + label_map[leaf_id]
        elif leaf_id in valid_id:
            directory = dir_path + 'validation/' + label_map[leaf_id]
        else:
            directory = dir_path + 'test'
        if not os.path.exists(directory):
            os.makedirs(directory)

        leaf_images[leaf_id].save(directory+'/' + name)

    if not cross_val:
        break

