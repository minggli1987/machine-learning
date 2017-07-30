# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score, silhouette_score
from sklearn.datasets import load_boston, load_iris, load_digits
from sklearn.manifold import TSNE


mnist = load_digits()['images']

iris, iris_target = load_iris()['data'], load_iris()['target']

db = KMeans(n_clusters=3)
iris_clusters = db.fit_predict(iris)
KMeans_iris = v_measure_score(iris_target, iris_clusters)
print(KMeans_iris)

gm = GaussianMixture(n_components=3)
gm.fit(iris)
gm_pred = gm.predict(iris)
gm_iris = v_measure_score(gm_pred, iris_target)
print(gm_iris)
print(gm.means_)
print(gm.covariances_)


print(np.isfinite(np.nan))
