# -*- coding: utf-8 -*-
"""
gp_test

play script with test data for running Gaussian Process in regression or
classification
"""
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as gaussian_kernel
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
kernel = gaussian_kernel(length_scale=1., length_scale_bounds=(1e-1, 1e3))
m = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

X, y = load_boston(return_X_y=True)
scores = cross_val_score(m, X, y, scoring='neg_mean_squared_error', cv=5)
print(scores)

m = ElasticNet()
scores = cross_val_score(m, X, y, scoring='neg_mean_squared_error', cv=5)
print(scores)
