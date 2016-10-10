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


# feature scaling using standard deviation as denominator

std_scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

# scaling

std_regressors = std_scaler.fit(regressors, regressand).transform(regressors)

# splitting

x_train, x_test, y_train, y_test = cross_validation.\
    train_test_split(std_regressors, regressand, test_size=.2)
reg = linear_model.LogisticRegression(fit_intercept=False, class_weight={0: 10, 1: 1})

# wrapping transformation and estimator into pipeline
pl = pipeline.Pipeline([('selected_scaler', std_scaler), ('logistic reg', reg)])

# pl.fit(x_train, y_train)
reg.fit(x_train, y_train)

# Stratified K-Fold

kf = cross_validation.StratifiedKFold(y_test, n_folds=5, shuffle=True)

# Model evaluation

cross_val_accuracy = cross_validation.cross_val_score(reg, x_test, y_test, scoring='accuracy', cv=kf)
roc_auc = cross_validation.cross_val_score(reg, x_test, y_test, scoring='roc_auc', cv=kf)
print(np.mean(roc_auc), np.mean(cross_val_accuracy))
accuracy = metrics.accuracy_score(y_test, reg.predict(x_test))
print(accuracy)
# Standard Gradient Descent

reg.coef_ = np.zeros(reg.coef_.shape)
optimiser = GradientDescent(alpha=.05, conv_thres=1e-8, display=False)
optimiser.fit(reg, x_train, y_train).optimise()
thetas, costs = optimiser.thetas, optimiser.costs

reg.coef_ = thetas  # applying optimised parameters

accuracy = metrics.accuracy_score(y_test, reg.predict(x_test))
print(accuracy)


def logit(X, theta):
    return 1 / (1 + np.exp(-np.dot(X, theta.T)))

result = logit(x_test, reg.coef_)
auto_result = reg.predict_proba(x_test)[:, 1]
