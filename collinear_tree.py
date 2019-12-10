from io import StringIO, BytesIO
from PIL import Image
import pydotplus

import numpy as np

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

data_dict = load_iris()
feature_names = data_dict["feature_names"]
target_names = data_dict["target_names"]
X, y = data_dict["data"], data_dict["target"]

collinar_feature = (X[:, 2] * 1.).reshape(-1, 1)
deficient_X = np.hstack([X[:, :], collinar_feature])

# X is now rank-deficient and therefore fails to span (k + 1)-dim hyperspace.
with np.testing.assert_raises(np.linalg.LinAlgError):
    np.linalg.cholesky(deficient_X)

clf = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1)
clf.fit(deficient_X, y)

dot_data = StringIO()
export_graphviz(
    clf,
    feature_names=feature_names + ["duplicated_" + feature_names[2]],
    class_names=target_names,
    filled=True,
    rounded=True,
    special_characters=True,
    out_file=dot_data
)

# sum of feature importances of collinear features are fixed and not the same
# for a single tree. as one feature is being used to gain information, the
# other highly collinear feature is unlikely to be used immediately after the
# previous split, instead feature C unrelated to either will be likely used.
# https://datascience.stackexchange.com/questions/31402/multicollinearity-in-decision-tree/33565
for _ in range(5):
    print(clf.feature_importances_[[2, -1]])
    clf.fit(deficient_X, y)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
with BytesIO() as f:
    f.write(graph.create_png())
    i = Image.open(f)
    i.show()

rf = RandomForestClassifier(
    n_estimators=10000,
    max_depth=None,
    min_samples_leaf=1)
rf.fit(deficient_X, y)
# asymptotically as n increases, the importances of both collinear features
# tend to the same value due to random sampling of feature space, sample space.
print("feature importances in random forest (10000 estimators) for two collinear features:")
print(rf.feature_importances_[[2, -1]])
