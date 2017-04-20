#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
helper

assistants with construction of neural network
"""

import tensorflow as tf
from itertools import cycle, tee


def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def train_nn(sess, x, y_, train_batch, test_batch, optimiser, n_epochs,
             r2, loss):
    """train neural network"""

    # require infinite test set looping for in-training validation
    test_batch = cycle(test_batch)
    train_batch_storage = tee(train_batch, n_epochs)
    train_errors = list()
    train_r2s = list()
    for epoch in range(n_epochs):
        step = 0
        for batch in train_batch_storage[epoch]:
            feat, target = zip(*batch)
            _, train_r2, train_error = sess.run([optimiser, r2, loss],
                                            feed_dict={x: feat, y_: target})
            step += 1
            if step % 5 == 0:
                batch = next(test_batch)
                feat, target = zip(*batch)
                test_r2, test_error = sess.run([r2, loss],
                                               feed_dict={x: feat, y_: target})
                print('epoch: {0:<3} step: {1:<3} '
                      'r2: {2:.4f} test error: {3:.6f}'.format(
                            epoch, step, test_r2, test_error))
            print('epoch: {0:<3} step: {1:<3} '
                  'r2: {2:.4f} train error: {3:.6f}'.format(
                         epoch, step, train_r2, train_error))
            train_errors.append(train_error)
            train_r2s.append(train_r2)
    return train_errors, train_r2s
