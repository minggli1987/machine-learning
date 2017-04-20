#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    from .nn.nn import train_errors, train_r2s

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.yaxis.tick_right()
    ax1.plot(list(range(len(train_errors))), train_errors, color='r')
    ax2.plot(list(range(len(train_r2s))), train_r2s)
    plt.show()
