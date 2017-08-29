# -*- coding: utf-8 -*-
"""

play programme for diabete regression problem using Gaussian Process.

"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

from xgboost import XGBRegressor


def dimension_reduction(X, k=.95):
    """
    reduce training data of n-dimension to k-dimension whilst retaining
    majority of variance. PCA minimises average project error.

    k must be integer or percentage of variance to retain.
    """
    assert isinstance(X, np.ndarray)
    m, n = X.shape

    pca = PCA(n_components=k)
    pca.fit(X)
    pct = pca.explained_variance_ratio_
    k = len(pct)
    pct = pct.sum()
    print("dataset of {0} dimensions has been reduced to {1} dimensions "
          "{2:.4%} variance retained.".format(n, k, pct))

    return pca.transform(X), k


def deserialize_score(json):
    """deserialize score dictionary"""
    json = json.copy()
    json.pop('fit_time', None)
    json.pop('score_time', None)
    return {k: v.mean() for k, v in json.items()}


np.random.seed(0)

# per documentation data has already been standardized i.e. (x - μ) / σ
data, target = load_diabetes(return_X_y=True)

_data = pd.DataFrame(
                data=data,
                columns=['var_{0}'.format(i) for i in range(data.shape[1])])
print(_data.describe())


# model selection cross validation
candidates = [
                LinearRegression(),
                Lasso(),
                Ridge(),
                ElasticNet(),
                RandomForestRegressor(n_estimators=100),
                XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100),
                GaussianProcessRegressor(alpha=1e-3)
]

cv_pipes = [make_pipeline(i) for i in candidates]
scores = ['r2', 'neg_mean_squared_error']
for pipe in cv_pipes:
    score_dict = cross_validate(pipe, data, target, cv=5, scoring=scores)
    print(deserialize_score(score_dict))


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.2)

# transforming training and test set separately.
reduced_X_train, k = dimension_reduction(X_train, k=0.9)
reduced_X_tests, _ = dimension_reduction(X_test, k=k)
