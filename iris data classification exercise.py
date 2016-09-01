from sklearn import metrics, cross_validation, naive_bayes, preprocessing, pipeline, linear_model
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

data = load_iris()

target_dict = {'species': {k: v for k, v in enumerate(data['target_names'])}}

df = pd.DataFrame(data['data'], columns=data['feature_names'], dtype=float) \
    .join(
    pd.DataFrame(data['target'], columns=['species'], dtype=int)).replace(target_dict)

print(target_dict)

df['species'] = pd.Categorical.from_array(df['species']).codes
df['species'] = df['species'].astype('category')

regressors = df[df.index[df.dtypes == float]]
regressand = df[df.index[df.dtypes == 'category']]

reg = linear_model.LogisticRegression()

kf_gen = cross_validation.KFold(df.shape[0], n_folds=5, shuffle=True)


prediction = cross_validation.cross_val_predict(reg, X=regressors, y=regressand, cv=kf_gen, n_jobs=-1)
df['pred'] = prediction
accuracy = cross_validation.cross_val_score(reg, regressors, regressand, scoring='accuracy', cv=kf_gen, n_jobs=-1)
print(accuracy)

with pd.ExcelWriter('iris_pred.xlsx') as writer:
    df.to_excel(writer, sheet_name='output', index=False)