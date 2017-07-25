# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans
from sklearn.metrics import v_measure_score, silhouette_score
from sklearn.datasets import load_boston, load_iris, load_digits
from sklearn.manifold import TSNE


mnist = load_digits()['images']

iris, iris_target = load_iris()['data'], load_iris()['target']

db = KMeans(n_clusters=3)
iris_clusters = db.fit_predict(iris)
KMeans_iris = v_measure_score(iris_target, iris_clusters)
print(iris_clusters)
print(iris_target)
print(KMeans_iris)


coefficientis = silhouette_score(iris, iris_clusters)
print(coefficientis)
