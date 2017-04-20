#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline

basic IO and feature engineering
"""

import pandas as pd
import numpy as np

from sklearn import model_selection, preprocessing


def load(path_to_data, dropna=True):
    """load csv and remove empty columns"""
    raw = pd.read_csv(filepath_or_buffer=path_to_data,
                      encoding='utf-8',
                      header=0,
                      dtype=np.float32
                      )
    # drop rows or columns if all elements are null
    return raw.dropna(axis=(0, 1), how='all', inplace=False) if dropna else raw


def transform(data, target, test_size=.2):
    """split and transform data for queue maker."""
    assert isinstance(data, pd.DataFrame), 'must require a Pandas DataFrame.'

    feature = np.array(data.ix[:, ~data.columns.isin([target])])
    target = np.array(data.ix[:, target])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                        feature, target, test_size=test_size)

    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    train_data = np.array([tuple((X_train_std[i], y_train[i]))
                          for i in range(len(X_train_std))])
    test_data = np.array([tuple((X_test_std[i], y_test[i]))
                          for i in range(len(X_test_std))])

    return train_data, test_data


def make_queue(data, batch_size=50, shuffle=True):
    """split by train, valid, and yield batches of required size."""

    n = len(data)
    # ceiling divison to round up to integer batch number.
    num_batches = -(-n // batch_size)

    for batch in range(num_batches):

        if shuffle:
            _data = np.random.permutation(data)
        elif not shuffle:
            _data = data

        start_index = batch * batch_size
        end_index = min(start_index + batch_size, n)
        if end_index <= start_index:
            break
        else:
            yield _data[start_index:end_index]
