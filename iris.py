import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import os
import pydotplus

from tensorflow.contrib import learn
from sklearn import (metrics, model_selection, naive_bayes, preprocessing,
                     linear_model, tree, svm)
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO
warnings.filterwarnings('ignore')

data = load_iris()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def visualize():

    iris = load_iris()

    X = iris.data  # we only take the first two features.
    Y = iris.target

    space = .5

    x_min, x_max = X[:, 0].min() - space, X[:, 0].max() + space
    y_min, y_max = X[:, 1].min() - space, X[:, 1].max() + space

    plt.figure(1, figsize=(12, 6))

    # Plot the training points
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.xlabel(iris['feature_names'][0])
    plt.ylabel(iris['feature_names'][1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks()
    plt.yticks()

    x_min, x_max = X[:, 2].min() - space, X[:, 2].max() + space
    y_min, y_max = X[:, 3].min() - space, X[:, 3].max() + space

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 2], X[:, 3], c=Y, cmap=plt.cm.Paired)
    plt.xlabel(iris['feature_names'][2])
    plt.ylabel(iris['feature_names'][3])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks()
    plt.yticks()

    plt.show()


def normalize(data):
    # building a scaler that applies to future data
    col_name = data.columns
    scaler = preprocessing.StandardScaler().fit(data)
    return pd.DataFrame(scaler.transform(data), columns=col_name)


# visualize()

target_dict = {'species': {k: v for k, v in enumerate(data['target_names'])}}

df = pd.DataFrame(
        data['data'], columns=data['feature_names'], dtype=np.float32).join(
     pd.DataFrame(
        data['target'], columns=['species'], dtype=np.int32)
        ).replace(target_dict)

df['species'] = df['species'].astype('category').cat.codes.astype('int32')
regressors = df.select_dtypes(include=['float32'])
regressand = df.select_dtypes(include=['int32'])


x_train, x_test, y_train, y_test = \
    model_selection.train_test_split(regressors, regressand, test_size=.2)

# Decision Tree classifier
reg = tree.DecisionTreeClassifier(max_depth=3,
                                  max_leaf_nodes=20,
                                  min_samples_leaf=15,
                                  random_state=2)
reg.fit(X=x_train, y=y_train)

kf_gen = model_selection.KFold(n_splits=5, shuffle=False, random_state=2)

prediction = reg.predict(x_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print('Single Tree Accuracy: {:.4f}'.format(accuracy))

# with pd.ExcelWriter('iris_pred.xlsx') as writer:
#     df.to_excel(writer, sheet_name='output', index=False)

dot_data = StringIO()
tree.export_graphviz(reg,
                     feature_names=regressors.columns,
                     class_names=target_dict['species'],
                     filled=True,
                     rounded=True,
                     special_characters=True,
                     out_file=dot_data)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

# random forest
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
prediction = rf.predict(x_test)
score = metrics.accuracy_score(y_test, prediction)
print('Random Forest Accuracy: {:.4f}'.format(score))


# logistic regression
lr = linear_model.LogisticRegression()
lr.fit(x_train, y_train)
prediction = lr.predict(x_test)
score = metrics.accuracy_score(y_test, prediction)
print('Logistic Regression Accuracy: {:.4f}'.format(score))

# Bayesian
clf = naive_bayes.GaussianNB()
clf.fit(x_train, y_train)
prediction = lr.predict(x_test)
score = metrics.accuracy_score(y_test, prediction)
print('Bayesian (GaussianNB) Accuracy: {:.4f}'.format(score))

# Gaussian Process classifier
gpc = GaussianProcessClassifier(kernel=RBF())
gpc.fit(x_train, y_train)
prediction = gpc.predict(x_test)
score = metrics.accuracy_score(y_test, prediction)
print('Gaussian Process Classsifier Accuracy: {:.4f}'.format(score))

# Support Vector Machine
svc = svm.SVC(C=1.0, kernel=RBF())
svc.fit(x_train, y_train)
prediction = svc.predict(x_test)
score = metrics.accuracy_score(y_test, prediction)
print('Support Vector Classsifier Accuracy: {:.4f}'.format(score))

# TensorFlow Neutral Network implementation


def main():

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    feature_columns = learn.infer_real_valued_columns_from_input(x_train)
    classifier = learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3)

    # Fit and predict.
    classifier.fit(x_train, y_train, steps=200)
    predictions = list(classifier.predict(x_test, as_iterable=True))
    score = metrics.accuracy_score(y_test, predictions)
    print('NN3 Accuracy: {:.4f}'.format(score))
    # print(y_test, predictions)


# if __name__ == '__main__':
#     main()
