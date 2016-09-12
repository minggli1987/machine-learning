from sklearn import metrics, cross_validation, naive_bayes, preprocessing, pipeline, linear_model, tree
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.externals.six import StringIO
import pydotplus

data = load_iris()

target_dict = {'species': {k: v for k, v in enumerate(data['target_names'])}}

df = pd.DataFrame(data['data'], columns=data['feature_names'], dtype=float) \
    .join(
    pd.DataFrame(data['target'], columns=['species'], dtype=int)).replace(target_dict)

df['species'] = pd.Categorical.from_array(df['species']).codes
df['species'] = df['species'].astype('category')

regressors = df.select_dtypes(include=['float'])
regressand = df.select_dtypes(include=['category'])

#print(df['species'][df['petal width (cm)'] > 0.8].value_counts())

reg = linear_model.LogisticRegression()
reg = naive_bayes.GaussianNB()
reg = tree.DecisionTreeClassifier(max_depth=3, max_leaf_nodes=20, min_samples_leaf=20, random_state=2)
reg.fit(X=regressors, y=regressand)

kf_gen = cross_validation.KFold(df.shape[0], n_folds=5, shuffle=False, random_state=2)

prediction = cross_validation.cross_val_predict(reg, X=regressors, y=regressand, cv=kf_gen, n_jobs=-1)
df['pred'] = prediction
accuracy = cross_validation.cross_val_score(reg, regressors, regressand, scoring='accuracy', cv=kf_gen, n_jobs=-1)
print(np.mean(accuracy))

# with pd.ExcelWriter('iris_pred.xlsx') as writer:
#     df.to_excel(writer, sheet_name='output', index=False)

dot_data = StringIO()
tree.export_graphviz(reg, feature_names=regressors.columns, class_names=target_dict['species'], filled=True, \
                     rounded=True, special_characters=True, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")