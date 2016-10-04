
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import preprocessing, cluster, metrics, cross_validation, linear_model, naive_bayes
from minglib import forward_select, backward_select
from GradientDescent import GradientDescent
from warnings import filterwarnings
filterwarnings('ignore')


def normalize(data):
    # building a scaler that applies to future data
    col_name = data.columns
    scaler = preprocessing.StandardScaler(with_mean=False, with_std=True).fit(data)
    return pd.DataFrame(scaler.transform(data), columns=col_name)

cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin', 'name']
data = pd.read_csv('data/auto-mpg.data-original.txt', header=None, delim_whitespace=True, names=cols)
data = data.dropna().reset_index(drop=True)
data['year'] = (1900 + data['year']).astype(int)
data.ix[:, :5] = data.ix[:, :5].astype(int)
data['origin'] = data['origin'].astype(int)


# selecting predictive variables
# regressors = data[[i for i in cols if i not in ['name']]]

# automatic stepwise (forward) selection
entire_numeric = data.select_dtypes(include=['int', 'float'])
fs_model, var = forward_select(normalize(entire_numeric), data[['acceleration']], display=False)
# print(var)
# print(fs_model.summary())
# selecting target variable
regressand = data['acceleration']
regressors = data[var]  # recommended variables from stepwise procedure

# splitting data - holdout validation
x_train, x_test, y_train, y_test = \
    cross_validation.train_test_split(np.column_stack((np.ones(regressors.shape[0]), normalize(regressors))),
                                      regressand, test_size=.3)
# splitting data - k fold cross validation
kf_gen = cross_validation.KFold(regressors.shape[0], n_folds=5, shuffle=True)

# fitting linear model
lr = linear_model.LinearRegression(fit_intercept=False)  # regressors already has constant 1
lr.fit(x_train, y_train)

init_mse = metrics.mean_squared_error(y_test, lr.predict(x_test))
init_r2 = metrics.r2_score(y_test, lr.predict(x_test))
print('the initial MSE currently stands at: {0:.2f}; '.format(init_mse), 'the initial R-squared stands at: {0:.2f}'.format(init_r2))


# preparing Gradient Descent

# generating initial parameters using the shape of existing ones
old_theta = lr.coef_
gd = GradientDescent(alpha=.005, max_epochs=5000, conv_thres=.000001, display=False)
gd.fit(x_train, y_train, lr)
gd.optimise()
new_theta, cost_set = gd.thetas, gd.costs
print(' old thetas are: ', old_theta, '\n', 'new thetas are: ', new_theta)

# applying new parameters
lr.coef_ = new_theta

# calculating new metrics
new_r2 = metrics.r2_score(y_test, lr.predict(x_test))
new_mse = metrics.mean_squared_error(y_test, lr.predict(x_test))

print('the new MSE currently stands at: {0:.2f}; '.format(new_mse), 'the R-squared stands at: {0:.2f}'.format(new_r2))

# k-fold generator intervals
# for train_index, test_index in kf_gen:
#
#     x_train, y_train = regressors.ix[train_index], regressand.ix[train_index]
#     x_test, y_test = regressors.ix[test_index], regressand.ix[test_index]
#
#     lr.fit(normalize(x_train), y_train)
#     predictions = lr.predict((normalize(x_test)))
#
#     # calculating MSE for linear model


kf_mse = cross_validation.cross_val_score\
    (lr, sm.add_constant(normalize(regressors)), regressand, scoring='mean_squared_error', cv=kf_gen)
kf_r2 = cross_validation.cross_val_score\
    (lr, sm.add_constant(normalize(regressors)), regressand, scoring='r2', cv=kf_gen)

print('the average MSE from k-fold validation: {0:.2f}; '.format(np.mean(abs(kf_mse))),
      'the average R-squared stands at: {0:.2f}'.format(np.mean(kf_r2)))

# fig, ax = plt.subplots()
# ax.scatter(data['horsepower'], data['acceleration'], color='b')
# ax.scatter(data['horsepower'], data['predicted_acceleration'], color='r')
# ax.set_xlabel('horsepower')
# ax.set_ylabel('acceleration')
# plt.show()

# multi-class Classifier on Origin

# transforming
dummy_cylinders = pd.get_dummies(data['cylinders'], prefix='cyl')
dummy_years = pd.get_dummies(data['year'], prefix='y')
dummies = pd.concat([dummy_cylinders, dummy_years], axis=1)
data = data.join(dummies)
cat_cols = [col for col in data.columns if col.startswith('y_') or col.startswith('cyl_')]

regressors = np.column_stack((np.ones(data.shape[0]), data[cat_cols]))
regressand = np.array(data['origin'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(regressors, regressand, test_size=.3)

sigmoid = linear_model.LogisticRegression(fit_intercept=False, class_weight='auto')
sigmoid.fit(x_train, y_train)

accuracy = metrics.accuracy_score(y_test, sigmoid.predict(x_test))
print('classier accuracy on testing stands at: {0:.2f}'.format(np.mean(accuracy)))

# gradient descent
old_theta = np.array(sigmoid.coef_)  # capturing parameters from logistic regression
sigmoid.coef_ = np.ones(old_theta.shape)  # generating initial parameters using the shape of existing ones
gd = GradientDescent(alpha=0.05, max_epochs=10000, display=False)
gd.fit(x_train, y_train, sigmoid)
gd.optimise()
new_theta, cost_set = gd.thetas, gd.costs

plt.plot(range(len(cost_set[2])), cost_set[2])
plt.show()
print(' old thetas are: ', [float('{0:.2f}'.format(i)) for i in old_theta[0]], '\n', 'new thetas are: ', [float('{0:.2f}'.format(i)) for i in new_theta[0]])

sigmoid.coef_ = new_theta

accuracy = metrics.accuracy_score(y_test, sigmoid.predict(x_test))
print('classier accuracy on testing stands at: {0:.2f}'.format(np.mean(accuracy)))


clf = naive_bayes.BernoulliNB()
clf.fit(x_train, y_train)
accuracy = metrics.accuracy_score(y_test, clf.predict(x_test))
print('classier accuracy on testing stands at: {0:.2f}'.format(np.mean(accuracy)))




# unique_origins = sorted(data['origin'].unique())
# testing_probs = pd.DataFrame(columns=unique_origins)
#
# models = dict()  # dict to contain different classifiers with each to produce one binomial output.
#
# for i in unique_origins:
#     classifier = linear_model.LogisticRegression(fit_intercept=False)
#     x = x_train
#     y = pd.DataFrame(y_train == i)
#
#     classifier.fit(x, y)
#
#     # gradient descent
#     old_theta = classifier.coef_  # capturing parameters from logistic regression
#     initial_params = np.ones(old_theta.T.shape)  # generating initial parameters using the shape of existing ones
#     new_theta, cost_set = gradient_descent(initial_params, x, y, classifier, alpha=0.01, max_epochs=10000)
#     classifier.coef_ = new_theta.T
#
#     models[i] = classifier
#     del classifier
#     testing_probs[i] = models[i].predict_proba(x_test)[:, 1]  # only capturing probability of positive result
#
#
#
# test_set = x_test.join(y_test).reset_index(drop=True)
# predictions = testing_probs.idxmax(axis=1)
# test_set['predicted_origin'] = predictions
# correct = test_set['origin'] == test_set['predicted_origin']
#
#
# # applying trained classifiers to full data
#
# probs = pd.DataFrame(columns=unique_origins)
#
# for i in unique_origins:
#     probs[i] = models[i].predict_proba(pd.DataFrame(np.column_stack((np.ones(data.shape[0]), data[cat_cols]))))[:, 1]
#
# data['predicted_origin'] = probs.idxmax(axis=1)
#
# print('classier accuracy on testing stands at: {0:.2f}'.format(len(test_set[correct]) / len(test_set)))
# print('classier accuracy on whole data stands at: {0:.2f}'.format(len(data[data['origin'] == data['predicted_origin']]) / len(data)))


# Clustering Cars
#
# data = data[[i for i in data.columns if (i not in dummies.columns) and (i not in ['cylinders', 'year'])]]
#
# plt.scatter(data['horsepower'], data['weight'])
# plt.show()
#
# clustering = cluster.KMeans(n_clusters=3, random_state=1).fit(data[['horsepower', 'weight']])
#
# data['cluster'] = clustering.labels_
#
# scatter_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#
# for n in range(len(set(clustering.labels_))):
#     clustered_data = data[data['cluster'] == n]
#     plt.scatter(clustered_data['horsepower'], clustered_data['weight'], color=scatter_colors[n])
# plt.show()
#
# legend = {1: 'North America', 2: 'Europe', 3: 'Asia'}
#
# for n in range(1, len(set(clustering.labels_)) + 1, 1):
#     clustered_data = data[data['origin'] == n]
#     plt.scatter(clustered_data['horsepower'], clustered_data['weight'], color=scatter_colors[n], label=legend[n])
# plt.legend(loc=2)
# plt.show()

