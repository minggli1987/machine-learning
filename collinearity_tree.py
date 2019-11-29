from io import StringIO, BytesIO
from PIL import Image
import pydotplus

import numpy as np

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(0)

data_dict = load_iris()
feature_names = data_dict["feature_names"]
target_names = data_dict["target_names"]
X, y = data_dict["data"], data_dict["target"]

collinar_feature = (X[:, 0] * 1.).reshape(-1, 1)
deficient_X = np.hstack([X, collinar_feature])

# X is now rank-deficient and therefore fails to span (k + 1)-dim hyperspace.
with np.testing.assert_raises(np.linalg.LinAlgError):
    np.linalg.cholesky(deficient_X)

clf = DecisionTreeClassifier(max_depth=5)
clf.fit(deficient_X, y)

dot_data = StringIO()
export_graphviz(clf,
                feature_names=feature_names + [feature_names[0]],
                class_names=target_names,
                filled=True,
                rounded=True,
                special_characters=True,
                out_file=dot_data)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
with BytesIO() as f:
    f.write(graph.create_png())
    i = Image.open(f)
    i.show()
