import os
import pandas as pd
import numpy as np
import warnings
from utils import delete_folders, extract, pic_resize
from GradientDescent import GradientDescent
from sklearn import metrics, model_selection, preprocessing
import tensorflow as tf
warnings.filterwarnings('ignore')

# coding: utf-8

__author__ = 'Ming Li'

"""This application forms a submission from Ming Li in regards to leaf kaggle challenge challenge on Kaggle community"""

# params

dir_path = 'leaf/images/'
id_label, id_name, mapping = extract('leaf/train.csv')
pic_names = [i.name for i in os.scandir(dir_path) if i.is_file()]
input_shape = (96, 96)
m = input_shape[0] * input_shape[1]
n = len(set(id_label.values()))

# cross validation of training photos

cross_val = False

kf_iterator = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=1)  # Stratified
train_x = list(id_name.keys())  # leaf id
train_y = list(id_name.values())  # leaf species names

for train_index, valid_index in kf_iterator.split(train_x, train_y):

    leaf_images = dict()  # temp dictionary of resized leaf images

    train_id = [train_x[i] for i in train_index]
    valid_id = [train_x[i] for i in valid_index]

    for name in pic_names:

        leaf_id = int(name.split('.')[0])
        leaf_images[leaf_id] = pic_resize(dir_path + name, size=input_shape, pad=True)

        if leaf_id in train_id:
            directory = dir_path + 'train/' + id_name[leaf_id]
        elif leaf_id in valid_id:
            directory = dir_path + 'validation/' + id_name[leaf_id]
        else:
            directory = dir_path + 'test'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)

        # leaf_images[leaf_id].save(directory+'/' + name)

    if not cross_val:
        break


# setting up tf Session

tf.device("/cpu:0")
sess = tf.Session()

# declare placeholders

x = tf.placeholder(dtype=tf.float32, shape=[None, m], name='feature')
y_ = tf.placeholder(dtype=tf.string, shape=[None, n], name='label')

# declare variables

# Variables
W = tf.Variable(tf.zeros([m, n]))
b = tf.Variable(tf.zeros([n]))

init = tf.global_variables_initializer()

y = tf.matmul(x, W) + b

# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


print(np.array(leaf_images[1]).flatten().shape)



# train
# for i in range(1000):
#     batch = mnist.train.next_batch(100)
#     train_step.run(feed_dict={x: batch[0], y_: batch[1]})


sess.run(init)
sess.close()

