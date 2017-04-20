#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nn

neural network module to predict power
"""
import tensorflow as tf

from .helper import weight_variable, bias_variable, train_nn
from ..pipeline import load, transform, make_queue

sess = tf.Session()

x = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='feature')
y_ = tf.placeholder(dtype=tf.float32, shape=[None], name='target')

with tf.name_scope('hidden_layer_1'):
    W_hidden1 = weight_variable([2, 20])
    b_hidden1 = bias_variable([20])

    h_hidden1 = tf.nn.relu(tf.matmul(x, W_hidden1) + b_hidden1)

with tf.name_scope('hidden_layer_2'):
    W_hidden2 = weight_variable([20, 20])
    b_hidden2 = bias_variable([20])

    h_hidden2 = tf.nn.relu(tf.matmul(h_hidden1, W_hidden2) + b_hidden2)

with tf.name_scope('output'):
    W_output = weight_variable([20, 1])
    b_output = bias_variable([1])

    h_output = tf.matmul(h_hidden2, W_output) + b_output

# minimize Mean Squared Error
cost_func = tf.reduce_mean(tf.square(tf.transpose(h_output) - y_))
# Coefficient of determination
tss = tf.reduce_sum(tf.square(tf.subtract(y_, tf.reduce_mean(y_))))
rss = tf.reduce_sum(tf.square(tf.subtract(y_, tf.transpose(h_output))))
r2 = tf.maximum(1 - tf.divide(rss, tss), 0)
# using Stochaistic Gradient Descent variant to find global minimum
train_step = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(cost_func)

init = tf.global_variables_initializer()
sess.run(init)

data = load('./data/dataset.csv', dropna=True)
train_data, test_data = transform(data, target='Power', test_size=.2)
train_batch, test_batch = make_queue(train_data), make_queue(test_data)

with sess:
    train_errors, train_r2s = train_nn(sess, x, y_, train_batch, test_batch,
                                       train_step, 500, r2, cost_func)
