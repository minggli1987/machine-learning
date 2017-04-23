#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import tensorflow as tf

from sklearn import model_selection
# for testing purpose, fixing random seed
random.seed(0)


def generate_training_data(total_size, seq_length):
    """generate sequence of 0s and 1s and label of count of 1s"""

    assert total_size / seq_length > 1, 'total sequence must be larger than ' \
                                        'individual length.'

    sequence_whole = random.choices(population=[0, 1],
                                    weights=(.5, .5),
                                    k=total_size)
    random.shuffle(sequence_whole)

    # ceiling division to get number of sub-sequences
    num_seq = -(-total_size // seq_length)

    sequence_nested = list()
    ones_count = list()
    for n in range(num_seq):
        index_start = n * seq_length
        index_end = min(total_size, (n + 1) * seq_length)
        sequence_sub = sequence_whole[index_start:index_end]
        sequence_nested.append(sequence_sub)
        ones_count.append(sequence_sub.count(1))

    return sequence_nested, ones_count


def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


train, label = generate_training_data(200, 20)

X_train, X_test, y_train, y_test = \
                model_selection.train_test_split(train,
                                                 label,
                                                 test_size=.2,
                                                 random_state=0)


x = tf.placeholder(dtype=tf.uint8, shape=[-1, 20, 1], name='feature')
y_ = tf.placeholder(dtype=tf.uint8, shape=[-1, 21], name='label')

num_hidden = 24

# define single lstm cell according to Hochreiter & Schmidhuber 1997
lstm_cell = tf.contrib.rnn.LSTMCell(num_units=num_hidden,
                                    activation=tf.tanh,
                                    use_peepholes=False)

# unrolled LSTM neural network, dynamnic_rnn or static_rnn
if True:
    outputs, final_state = tf.contrib.rnn.static_rnn(cell=lstm_cell, data=x)
if False:
    outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cell, data=x)

print(outputs.get_shape())

outputs_transposed = tf.transpose(outputs, [1, 0, 2])
last = tf.gather(outputs_transposed, int(outputs_transposed.get_shape()[0]) - 1)

weight = weight_variable(shape=[num_hidden, y_.get_shape()[1]])
bias = bias_variable(shape=[y_.get_shape()[1]])

logits = tf.matmul(last, weight) + bias

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(loss)

# eval
mistakes = tf.not_equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
error_rate = tf.reduce_mean(tf.cast(mistakes, tf.float32))


init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)


batch_size = 1000
no_of_batches = int(len(X_train)/batch_size)
epoch = 5000
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = X_train[ptr:ptr+batch_size], y_train[ptr:ptr+batch_size]
        ptr += batch_size
        sess.run(train_step, {x: inp, y_: out})
    print("Epoch - ", str(i))
incorrect = sess.run(error_rate, feed_dict={x: X_test, y_: y_test})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()
