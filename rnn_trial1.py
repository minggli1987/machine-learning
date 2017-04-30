#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import tensorflow as tf
import numpy as np

from sklearn import model_selection

# for testing purpose, fixing random seed for determinisim
random.seed(0)

N_CLASS = 21
STATE_SIZE = 24
STEP_SIZE = 20
BATCH_SIZE = 1000


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
    labels = list()
    for n in range(num_seq):
        index_start = n * seq_length
        index_end = min(total_size, (n + 1) * seq_length)
        sequence_sub = sequence_whole[index_start:index_end]
        sequence_nested.append(sequence_sub)
        labels.append(sequence_sub.count(1))
    # reshape training data to (inferred row count, 1, sequence length)
    train = np.array(sequence_nested).reshape(-1, 1, seq_length)
    # one-hot label using eye if labels can be used as numerical index.
    one_hot_label = np.eye(seq_length + 1)[np.array(labels)]
    return train, one_hot_label


def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


train, label = generate_training_data(int(1e6), STEP_SIZE)

X_train, X_test, y_train, y_test = \
                model_selection.train_test_split(train,
                                                 label,
                                                 test_size=.2,
                                                 random_state=0)

x = tf.placeholder(dtype=tf.float32,
                   shape=[None, 1, STEP_SIZE],
                   name='feature')
y_ = tf.placeholder(dtype=tf.uint8,
                    shape=[None, N_CLASS],
                    name='label')
init_state = tf.zeros(shape=[BATCH_SIZE, STATE_SIZE], name='initial_state')
init_output = tf.zeros(shape=[BATCH_SIZE, STATE_SIZE], name='initial_output')


def rnn_params_initializer():
    with tf.variable_scope('rnn_cell'):
        W_hx = tf.get_variable(
                    name='W_hx',
                    shape=[STEP_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        W_hh = tf.get_variable(
                    name='W_hh',
                    shape=[STATE_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        b_h = tf.get_variable(
                    name='b_h',
                    shape=[STATE_SIZE],
                    initializer=tf.constant_initializer(0.0)
                    )


def rnn_cell(rnn_input, state):
    """implementation according to Lipton et al (2015) with sigmoid function
    replaced by hyberbolic tangent."""
    with tf.variable_scope('rnn_cell', reuse=True):
        W_hx = tf.get_variable(
                    name='W_hx',
                    shape=[STEP_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        W_hh = tf.get_variable(
                    name='W_hh',
                    shape=[STATE_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        b_h = tf.get_variable(
                    name='b_h',
                    shape=[STATE_SIZE],
                    initializer=tf.constant_initializer(0.0)
                    )
    return tf.tanh(tf.matmul(rnn_input, W_hx) + tf.matmul(state, W_hh) + b_h)


def lstm_params_initializer():
    with tf.variable_scope('lstm_cell'):
        forget_W_hx = tf.get_variable(
                    name='forget_W_hx',
                    shape=[STEP_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        forget_W_hh = tf.get_variable(
                    name='forget_W_hh',
                    shape=[STATE_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        forget_b_h = tf.get_variable(
                    name='forget_b_h',
                    shape=[STATE_SIZE],
                    initializer=tf.constant_initializer(0.0)
                    )
        input_W_hx = tf.get_variable(
                    name='input_W_hx',
                    shape=[STEP_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        input_W_hh = tf.get_variable(
                    name='input_W_hh',
                    shape=[STATE_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        input_b_h = tf.get_variable(
                    name='input_b_h',
                    shape=[STATE_SIZE],
                    initializer=tf.constant_initializer(0.0)
                    )
        cell_state_W_hx = tf.get_variable(
                    name='cell_state_W_hx',
                    shape=[STEP_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        cell_state_W_hh = tf.get_variable(
                    name='cell_state_W_hh',
                    shape=[STATE_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        cell_state_b_h = tf.get_variable(
                    name='cell_state_b_h',
                    shape=[STATE_SIZE],
                    initializer=tf.constant_initializer(0.0)
                    )
        output_W_hx = tf.get_variable(
                    name='output_W_hx',
                    shape=[STEP_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        output_W_hh = tf.get_variable(
                    name='output_W_hh',
                    shape=[STATE_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        output_b_h = tf.get_variable(
                    name='output_b_h',
                    shape=[STATE_SIZE],
                    initializer=tf.constant_initializer(0.0)
                    )


def lstm_cell(lstm_input, lstm_output, cell_state):
    with tf.variable_scope('lstm_cell', reuse=True):
        forget_W_hx = tf.get_variable(
                    name='forget_W_hx',
                    shape=[STEP_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        forget_W_hh = tf.get_variable(
                    name='forget_W_hh',
                    shape=[STATE_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        forget_b_h = tf.get_variable(
                    name='forget_b_h',
                    shape=[STATE_SIZE],
                    initializer=tf.constant_initializer(0.0)
                    )
        input_W_hx = tf.get_variable(
                    name='input_W_hx',
                    shape=[STEP_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        input_W_hh = tf.get_variable(
                    name='input_W_hh',
                    shape=[STATE_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        input_b_h = tf.get_variable(
                    name='input_b_h',
                    shape=[STATE_SIZE],
                    initializer=tf.constant_initializer(0.0)
                    )
        cell_state_W_hx = tf.get_variable(
                    name='cell_state_W_hx',
                    shape=[STEP_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        cell_state_W_hh = tf.get_variable(
                    name='cell_state_W_hh',
                    shape=[STATE_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        cell_state_b_h = tf.get_variable(
                    name='cell_state_b_h',
                    shape=[STATE_SIZE],
                    initializer=tf.constant_initializer(0.0)
                    )
        output_W_hx = tf.get_variable(
                    name='output_W_hx',
                    shape=[STEP_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        output_W_hh = tf.get_variable(
                    name='output_W_hh',
                    shape=[STATE_SIZE, STATE_SIZE],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )
        output_b_h = tf.get_variable(
                    name='output_b_h',
                    shape=[STATE_SIZE],
                    initializer=tf.constant_initializer(0.0)
                    )
        forget_gate = tf.sigmoid(
                      tf.matmul(lstm_input, forget_W_hx) +
                      tf.matmul(lstm_output, forget_W_hh) +
                      forget_b_h
                      )
        input_gate = tf.sigmoid(
                     tf.matmul(lstm_input, input_W_hx) +
                     tf.matmul(lstm_output, input_W_hh) +
                     input_b_h
                     )
        cell_state_delta = tf.tanh(
                           tf.matmul(lstm_input, cell_state_W_hx) +
                           tf.matmul(lstm_output, cell_state_W_hh) +
                           cell_state_b_h
                           )
        # cell memory forgets old information and learns new information
        cell_state_t = forget_gate * cell_state + input_gate * cell_state_delta

        output_gate = tf.sigmoid(
                      tf.matmul(lstm_input, output_W_hx) +
                      tf.matmul(lstm_output, output_W_hh) +
                      output_b_h
                      )

        return output_gate * tf.tanh(cell_state_t), cell_state_t


rnn_inputs = tf.unstack(x, axis=1)
# static rnn unrolled
state = init_state
output = init_output
lstm_params_initializer()
rnn_outputs = list()
for rnn_input in rnn_inputs:
    output, state = lstm_cell(rnn_input, output, state)
    rnn_outputs.append(state)

final_state = rnn_outputs[-1]

# weights in shape (24, 21) and bias in shape (21, ), for softmax
W_yh = weight_variable(shape=[STATE_SIZE, N_CLASS])
b_y = bias_variable(shape=[N_CLASS])

# logits in (batch_size, 21)
logits = tf.matmul(final_state, W_yh) + b_y

# softmax_cross_entropy_with_logits to produce numerically robust Xent
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_)

loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss)

# eval
mistakes = tf.not_equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
error_rate = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)


no_of_batches = int(len(X_train)/BATCH_SIZE)
epoch = 200
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = X_train[ptr:ptr+BATCH_SIZE], y_train[ptr:ptr+BATCH_SIZE]
        ptr += BATCH_SIZE
        sess.run(train_step, {x: inp, y_: out})
    print("Epoch - ", str(i))

no_of_batches = int(len(X_test)/BATCH_SIZE)
incorrect_rates = list()
ptr = 0
for j in range(no_of_batches):
    inp, out = X_test[ptr:ptr+BATCH_SIZE], y_test[ptr:ptr+BATCH_SIZE]
    ptr += BATCH_SIZE
    incorrect_rates.append(sess.run(error_rate, feed_dict={x: inp, y_: out}))
incorrect = np.array(incorrect_rates).mean()
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()
