import numpy as np
from GradientDescent import GradientDescent
import pandas as pd
from sklearn import metrics, cross_validation, naive_bayes, preprocessing, pipeline, linear_model, tree, \
    decomposition, pipeline
import matplotlib.pyplot as plt

__author__ = 'Ming Li'

# Loading source data into

source_path = 'data/pima-diabetes.csv'

data_type = {
    'Pregnancies': np.int,
    'BMI': np.float,
    'DiabetesPedigreeFunction': np.float,
    'Glucose': np.int,
    'BloodPressure': np.int,
    'SkinThickness': np.int,
    'Insulin': np.int,
    'Age': np.int,
    'Outcome': np.bool_
}

data = pd.read_csv(filepath_or_buffer=source_path, dtype=data_type)

regressand = np.ravel(data.select_dtypes(include=[np.bool_]))
regressors = np.array(data.select_dtypes(exclude=[np.bool_]))
regressors = np.column_stack((np.ones(regressors.shape[0]), regressors))

# splitting

x_train, x_test, y_train, y_test = cross_validation.\
    train_test_split(regressors, regressand, test_size=.2, random_state=1)

# feature scaling using standard deviation as denominator

std_scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

model_dict = [
    linear_model.LogisticRegression(fit_intercept=False, class_weight={0: 10, 1: 1}),
    naive_bayes.GaussianNB()
]

# wrapping transformation and estimator into pipeline
pl = pipeline.Pipeline([('selected_scaler', std_scaler), ('logistic reg', model_dict[0])])
# pl.fit(x_train, y_train)
model_dict[0].fit(x_train, y_train)
# Stratified K-Fold

kf = cross_validation.StratifiedKFold(y_test, n_folds=5, shuffle=False, random_state=1)


# Model evaluation

accuracy = cross_validation.cross_val_score(pl, x_test, y_test, scoring='accuracy', cv=kf)
roc_auc = cross_validation.cross_val_score(pl, x_test, y_test, scoring='roc_auc', cv=kf)
print(np.mean(roc_auc), np.mean(accuracy))

parameters = model_dict[0].coef_
print(parameters.shape)

linear_model.LinearRegression()

def logit(X, theta):
    return 1 / (1 + np.exp(-np.dot(X, theta.T)))

result = logit(x_test, parameters)
auto_result = model_dict[0].predict_proba(x_test)[:, 1]

print(result.shape)