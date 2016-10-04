import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import preprocessing, cluster, metrics, cross_validation, linear_model, feature_selection, naive_bayes
from minglib import forward_select, backward_select
from GradientDescent import GradientDescent


# cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin', 'name']
# data = pd.read_csv('data/auto-mpg.data-original.txt', header=None, delim_whitespace=True, names=cols)
# data = data.dropna().reset_index(drop=True)
# data['year'] = (1900 + data['year']).astype(int)
# data.ix[:, :5] = data.ix[:, :5].astype(int)
# data['origin'] = data['origin'].astype(int)

df_math = pd.read_csv('data/student-mat.csv', delimiter=';', header=0)
df_por = pd.read_csv('data/student-por.csv', delimiter=';', header=0)

df = df_math.copy()


def standardize(data):
    cols = data.columns
    scaler = preprocessing.StandardScaler().fit(data)
    return pd.DataFrame(scaler.transform(data), columns=cols)


def normalize(data):
    cols = data.columns
    scaler = preprocessing.Normalizer().fit(data)
    return pd.DataFrame(scaler.transform(data), columns=cols)


def nonscaled(data):
    cols = data.columns
    return pd.DataFrame(data, columns=cols)


def categorical_to_code(var):
    return pd.Categorical.from_array(var).codes


'''
1 school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
2 sex - student's sex (binary: 'F' - female or 'M' - male)
3 age - student's age (numeric: from 15 to 22)
4 address - student's home address type (binary: 'U' - urban or 'R' - rural)
5 famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
6 Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
7 Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
8 Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
9 Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
10 Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
11 reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
12 guardian - student's guardian (nominal: 'mother', 'father' or 'other')
13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
15 failures - number of past class failures (numeric: n if 1<=n<3, else 4)
16 schoolsup - extra educational support (binary: yes or no)
17 famsup - family educational support (binary: yes or no)
18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
19 activities - extra-curricular activities (binary: yes or no)
20 nursery - attended nursery school (binary: yes or no)
21 higher - wants to take higher education (binary: yes or no)
22 internet - Internet access at home (binary: yes or no)
23 romantic - with a romantic relationship (binary: yes or no)
24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
25 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
26 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
29 health - current health status (numeric: from 1 - very bad to 5 - very good)
30 absences - number of school absences (numeric: from 0 to 93)
'''


obj_col = df.dtypes[df.dtypes == 'object'].index  # locate all object columns to be transformed

df[obj_col] = df[obj_col].apply(categorical_to_code, axis=0)  # transforming

# metadata for data set
data_mapping = dict()
for i in df_math.columns:
    data_mapping[i] = df[i].unique().shape[0]

# looking for multi-nominal categorical variables that require dummy coding
# print({k: v for k, v in data_mapping.items() if (v < 5) and (v > 2)})

# transforming non-linear multi-nominal categorical variable
reason_dummies = pd.get_dummies(df['reason'], prefix='reason',)
guardian_dummies = pd.get_dummies(df['guardian'], prefix='guardian')
df.drop(df[['reason', 'guardian']], axis=1, inplace=True)
df = df.join(reason_dummies).join(guardian_dummies)


# PREDICTING CONTINUOUS VARIABLE

lr = linear_model.LinearRegression(fit_intercept=False)

target = input('please input target variable name: ')
regressand = df[target]
regressors = df[[i for i in df.columns if i != target]]  # if i not in ['G1', 'G2']]]

# How variables are to be rescaled
scaler = standardize

# stepwise backward selection
# sw_model, var = forward_select(scaler(regressors), regressand, alpha=0.05, display=False)
rfe = feature_selection.RFE(lr)
rfe = rfe.fit(scaler(regressors), regressand)
var = regressors.columns[rfe.support_]
regressors = regressors[var]
print(var)


# holdout validation

x_train, x_test, y_train, y_test = \
    cross_validation.train_test_split(sm.add_constant(scaler(regressors)), regressand, test_size=.2, random_state=1)

lr.fit(x_train, y_train)

gd = GradientDescent(alpha=.1, max_epochs=5000, conv_thres=.0001, display=True).fit(x_train, y_train, lr)
gd.optimise()
new_theta, costs = gd.thetas, gd.costs
# plt.plot(range(len(costs)), costs)
# plt.show()


predicted = lr.predict(x_test)
mse = metrics.mean_squared_error(y_test, predicted)
r2 = metrics.r2_score(y_test, predicted)
print(mse, r2)

# plt.plot(df.index, df['G3'], label='Final Score', color='b')
# plt.plot(df.index, df['pred_G3'], label='Pred Score', color='r')
# plt.legend(loc=2)
# plt.show()

kf_gen = cross_validation.KFold(regressors.shape[0], n_folds=10, shuffle=True, random_state=5)
kf_mse = cross_validation.cross_val_score\
    (lr, sm.add_constant(scaler(regressors)), regressand, scoring='mean_squared_error', cv=kf_gen)
kf_r2 = cross_validation.cross_val_score\
    (lr, sm.add_constant(scaler(regressors)), regressand, scoring='r2', cv=kf_gen)

print('the average MSE from k-fold validation: {0:.2f}; '.format(np.mean(abs(kf_mse))))
print('the average R-squared stands at: {0:.2f}'.format(np.mean(kf_r2)))


# PREDICTING ROMANTIC RELATIONSHIP AT SCHOOL

locr = linear_model.LogisticRegression()

target = input('please input target variable name: ')
regressand = df[target]
regressors = df[[i for i in df.columns if i != target]]

# Scaler

scaler = nonscaled

# Feature Seletion

rfe = feature_selection.RFE(locr)
rfe = rfe.fit(scaler(regressors), regressand)
var = regressors.columns[rfe.support_]
regressors = regressors[var]
print(var)

# HOLDOUT

x_train, x_test, y_train, y_test = \
    cross_validation.train_test_split(scaler(regressors), regressand, test_size=.2, random_state=9)

locr.fit(x_train, y_train)
gd = GradientDescent(display=True).fit(x_train, y_train, locr)
gd.optimise()
new_theta, costs = gd.thetas, gd.costs
# plt.plot(range(len(costs)), costs)
# plt.show()
print(locr.coef_.shape, new_theta.shape)

predicted = locr.predict(x_test)
roc_auc = metrics.roc_auc_score(y_test, predicted)

test = x_test.join(y_test)
test['predicted'] = predicted
condition = test[target] == test['predicted']
accuracy = len(test[condition]) / len(test)
print('Accuracy: {0:.2f}'.format(accuracy))
print('ROC Area Under Curve: {0:.2f}'.format(roc_auc))
print('True Positive Rate: {0:.2f}'.format(len(test[(test['predicted'] == 1) & (test[target] == 1)]) / len(test[test[target] == 1])))
print('True Negative Rate: {0:.2f}'.format(len(test[(test['predicted'] == 0) & (test[target] == 0)]) / len(test[test[target] == 0])))
print('False Positive Rate: {0:.2f}'.format(len(test[(test['predicted'] == 1) & (test[target] == 0)]) / len(test[test[target] == 0])))
print('False Negative Rate: {0:.2f}'.format(len(test[(test['predicted'] == 0) & (test[target] == 1)]) / len(test[test[target] == 1])))


kf_gen = cross_validation.KFold(regressors.shape[0], n_folds=10, shuffle=True, random_state=5)
kf_rocauc = cross_validation.cross_val_score\
    (locr, scaler(regressors), regressand, scoring='roc_auc', cv=kf_gen)
kf_accuracy = cross_validation.cross_val_score\
    (locr, scaler(regressors), regressand, scoring='accuracy', cv=kf_gen)

print('the average ROC AUC from k-fold validation: {0:.2f}; '.format(np.mean(abs(kf_rocauc))),
      'the average Accuracy stands at: {0:.2f}'.format(np.mean(kf_accuracy)))