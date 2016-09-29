__author__ = 'Ming Li'

# This application forms a submission from Ming in regards to leaf classification challenge on Kaggle community

import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import metrics, cross_validation, naive_bayes, preprocessing, pipeline, linear_model, tree, decomposition
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from sklearn.externals.six import StringIO
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from minglib import gradient_descent
import pydotplus
warnings.filterwarnings('ignore')


test = pd.read_csv('data/leaf/test.csv')
train = pd.read_csv('data/leaf/train.csv')

print(train.info())