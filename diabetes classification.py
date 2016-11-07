import numpy as np
from GradientDescent import GradientDescent
import pandas as pd
from sklearn import metrics, model_selection, naive_bayes, preprocessing, pipeline, linear_model, tree, \
    decomposition, pipeline
import matplotlib.pyplot as plt
import seaborn as sns

__author__ = 'Ming Li'

# loading source data into pandas

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

for column in data.columns:
    if column not in ['Outcome']:
        data[column] = data[column].apply(lambda x: np.nan if x == 0 else x)

complete_rows = data.dropna(how='any', axis=0, inplace=False).index.tolist()
data = data.ix[data.index.isin(complete_rows)].reset_index(drop=True)

# PCA

print(data.corr()['Outcome'].sort_values(ascending=False))  # correlation with Diabete Diagnosis outcome

# visualising Principal Components

x, x_label = data['Age'], data['Age'].name
y, y_label = data['Glucose'], data['Glucose'].name

plt.scatter(x, y, c=data['Outcome'], cmap=plt.cm.Paired)

plt.title('Diabetes in Pima, India')
plt.ylim((y.min() - 5, y.max() + 5))
plt.ylabel(y_label)
plt.xlim((x.min() - 5, x.max() + 5))
plt.xlabel(x_label)
plt.show()

# seperating predictive variables and target

regressand = np.ravel(data.select_dtypes(include=[np.bool_])).copy()
regressors = np.array(data.select_dtypes(exclude=[np.bool_])).copy()
regressors = np.column_stack((np.ones(regressors.shape[0]), regressors))

# feature scaling using standard deviation as denominator

std_scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

# scaling

std_regressors = std_scaler.fit(regressors, regressand).transform(regressors)

# splitting

x_train, x_test, y_train, y_test = model_selection.\
    train_test_split(std_regressors, regressand, test_size=.2)
reg = linear_model.LogisticRegression(fit_intercept=False, class_weight={0: 10, 1: 1})

# wrapping transformation and estimator into pipeline
pl = pipeline.Pipeline([('selected_scaler', std_scaler), ('logistic reg', reg)])

# pl.fit(x_train, y_train)
reg.fit(x_train, y_train)

# Stratified K-Fold

kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True)

# Model evaluation

cross_val_accuracy = model_selection.cross_val_score(reg, x_test, y_test, scoring='accuracy', cv=kf)
roc_auc = model_selection.cross_val_score(reg, x_test, y_test, scoring='roc_auc', cv=kf)
print('{0:.2f}'.format(np.mean(roc_auc)), '{0:.2f}'.format(np.mean(cross_val_accuracy)))
accuracy = metrics.accuracy_score(y_test, reg.predict(x_test))
print('Initial model accuracy: {0:.2f}'.format(accuracy))

# Standard Gradient Descent

reg.coef_ = np.zeros(reg.coef_.shape)
optimiser = GradientDescent(alpha=.01, conv_thres=1e-8, display=False)
optimiser.fit(reg, x_train, y_train).optimise()
thetas, costs = optimiser.thetas, optimiser.costs

reg.coef_ = thetas  # applying optimised parameters

plt.plot(range(len(costs)), costs)
plt.show()

accuracy = metrics.accuracy_score(y_test, reg.predict(x_test))
print('Final model accuracy: {0:.2f}'.format(accuracy))



def linear(X, theta):
    return np.dot(X, theta.T)


def logit(X, theta):
    return 1 / (1 + np.exp(-linear(X, theta)))


result = np.ravel(logit(x_test, reg.coef_))
auto_result = reg.predict_proba(x_test)[:, 1]

