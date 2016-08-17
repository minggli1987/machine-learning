import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import preprocessing, cluster, metrics, cross_validation, linear_model
from minglib import forward_selected, gradient_descent


# cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin', 'name']
# data = pd.read_csv('data/auto-mpg.data-original.txt', header=None, delim_whitespace=True, names=cols)
# data = data.dropna().reset_index(drop=True)
# data['year'] = (1900 + data['year']).astype(int)
# data.ix[:, :5] = data.ix[:, :5].astype(int)
# data['origin'] = data['origin'].astype(int)

df_math = pd.read_csv('data/student-mat.csv', delimiter=';', header=0)
df_por = pd.read_csv('data/student-por.csv', delimiter=';', header=0)

print(df_math)