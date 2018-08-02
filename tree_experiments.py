import numpy as np

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(0)

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_hat = rf.predict(X_test)
print(accuracy_score(y_test, y_hat))


scaler = StandardScaler()
pip = Pipeline([('std', scaler), ('rf', rf)])
pip.fit(X_train, y_train)
y_hat = pip.predict(X_test)
print(accuracy_score(y_test, y_hat))


scaler = Normalizer()
pip = Pipeline([('normalizer', scaler), ('rf', rf)])
pip.fit(X_train, y_train)
y_hat = pip.predict(X_test)
print(accuracy_score(y_test, y_hat))


scaler = StandardScaler()
pca = PCA()
pip = Pipeline([('std', scaler), ('pca', pca), ('rf', rf)])
pip.fit(X_train, y_train)
y_hat = pip.predict(X_test)
print(accuracy_score(y_test, y_hat))

