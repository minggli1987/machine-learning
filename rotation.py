"""
rotation axes

how decision tree handles rotated linear decision boundary
"""
from io import StringIO, BytesIO

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from PIL import Image
import pydotplus

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.style.use('ggplot')

np.random.seed(0)


def rotate(degree):
    """produce rotation matrix anti-clockwise by x degree."""
    r = np.radians(degree)
    return np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])


x = np.random.rand(100, 1)
y = np.random.rand(100, 1)
# diagonal artificial decision boundary
z = x < y

data = np.concatenate((x, y, z), axis=1)
df = pd.DataFrame(data, columns=['x', 'y', 'label'])

sns.lmplot('x', 'y', df, hue='label', fit_reg=False)
plt.show()

clf = DecisionTreeClassifier()

clf.fit(df[['x', 'y']], df['label'])

dot_data = StringIO()
export_graphviz(clf,
                feature_names=['x', 'y'],
                class_names=df['label'].astype(str),
                filled=True,
                rounded=True,
                special_characters=True,
                out_file=dot_data)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
with BytesIO() as f:
    f.write(graph.create_png())
    i = Image.open(f)
    i.show()

rotation_matrix = rotate(45)
rotated = np.concatenate((data[:, :2].dot(rotation_matrix),
                          data[:, -1].reshape(-1, 1)), axis=1)
df = pd.DataFrame(rotated, columns=['x', 'y', 'label'])
sns.lmplot('x', 'y', df, hue='label', fit_reg=False)
plt.show()

clf.fit(df[['x', 'y']], df['label'])

dot_data = StringIO()
export_graphviz(clf,
                feature_names=['x', 'y'],
                class_names=df['label'].astype(str),
                filled=True,
                rounded=True,
                special_characters=True,
                out_file=dot_data)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
with BytesIO() as f:
    f.write(graph.create_png())
    i = Image.open(f)
    i.show()
