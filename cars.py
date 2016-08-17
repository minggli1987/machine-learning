
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import preprocessing, cluster, metrics, cross_validation, linear_model
from scipy.stats import t


# class NumFmt(object):
#     def __init__(self, n):
#         self.value = float(n)
#
#     def converted(self):
#         return '{0:.2f}'.format(self.value)
#
#     def __eq__(self, other):
#         return self.value == other.value
#
#     def __lt__(self, other):
#         return self.value < other.value

# cost function, partial_derivative and gradient descent algorithm


def cost_function(params, x, y):
    params = np.array(params)
    x = np.array(x)
    y = np.array(y)
    J = 0
    m = len(x)
    for i in range(m):
        h = np.sum(params.T * x[i])
        diff = (h - y[i]) ** 2
        J += diff
    J /= (2 * m)
    return J


def partial_derivative_cost(params, j, x, y):
    params = np.array(params)
    x = np.array(x)
    y = np.array(y)
    J = 0
    m = len(x)
    for i in range(m):
        h = np.sum(params.T * x[i])
        diff = (h - y[i]) * x[i][j]
        J += diff
    J /= m
    return J


def gradient_descent(params, x, y, alpha=0.1):
    max_epochs = 10000  # max number of iterations
    count = 0  # initiating a count number so once reaching max iterations will terminate
    conv_thres = 0.000001  # convergence threshold

    cost = cost_function(params, x, y)  # convergence threshold

    prev_cost = cost + 10
    costs = [cost]
    thetas = [params]

    #  beginning gradient_descent iterations

    print('\nbeginning gradient decent algorithm...\n')

    while (np.abs(prev_cost - cost) > conv_thres) and (count <= max_epochs):
        prev_cost = cost
        update = np.zeros(len(params))  # simultaneously update all thetas

        for j in range(len(params)):
            update[j] = alpha * partial_derivative_cost(params, j, x, y)

        params -= update  # descending

        thetas.append(params)  # restoring historic parameters

        cost = cost_function(params, x, y)

        costs.append(cost)
        count += 1

    return params, costs


def forward_selected(data, target):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response w

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """

    remaining = set(data.columns)
    remaining.remove(target)
    selected_var = []
    current_score, best_new_score = 0.0, 0.0
    alpha = 0.05

    print('\nbeginning forward stepwise variable selection...\n')
    while remaining and current_score == best_new_score:

        scores_with_candidates = []  # containing variables

        for candidate in remaining:

            X = sm.add_constant(data[selected_var + [candidate]])  # inherit variables from last step and try new ones
            reg = sm.OLS(data[target], X).fit()
            score, p = reg.rsquared, reg.pvalues[-1]  # r2 (changeable) and two-tailed p value of the candidate
            scores_with_candidates.append((score, p, candidate))

        scores_with_candidates.sort(reverse=False)  # order variables by score in ascending
        disqualified_candidates = [i for i in scores_with_candidates if ~(i[1] < alpha)]
        scores_with_candidates = [i for i in scores_with_candidates if i[1] < alpha]
        try:
            best_new_score, best_candidate_p, best_candidate = scores_with_candidates.pop()
        except IndexError:  # when no candidate passes significance test at critical value
            disqualified_score, disqualified_p, disqualified_candidate = disqualified_candidates.pop(0)
            remaining.remove(disqualified_candidate)  # remove the worst disqualified candidate
#            print(disqualified_score, disqualified_p, disqualified_candidate)
            continue  # continuing the while loop
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            current_score = best_new_score
            selected_var.append(best_candidate)
#            print(best_new_score, best_candidate_p, best_candidate)

    model = sm.OLS(data[target], sm.add_constant(data[selected_var])).fit()
    return model, selected_var


def numfmt(num):
    assert isinstance(num, (int, float))
    return float('{0:.2f}'.format(num))


def normalize(data):
    # building a scaler that applies to future data
    col_name = data.columns
    scaler = preprocessing.StandardScaler().fit(data)
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
entire_numeric = data[[i for i in cols if i not in ['name']]]
fs_model, var = forward_selected(normalize(entire_numeric), 'acceleration')

# selecting target variable
regressand = data['acceleration']
regressors = data[var]  # recommended variables from stepwise procedure

# splitting data - holdout validation
x_train, x_test, y_train, y_test = \
    cross_validation.train_test_split(sm.add_constant(normalize(regressors)), regressand, test_size=.2)
# splitting data - k fold cross validation
kf_gen = cross_validation.KFold(regressors.shape[0], n_folds=10, shuffle=True)

# fitting linear model
lr = linear_model.LinearRegression(fit_intercept=False)  # regressors already has intercept manually
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)

init_mse = numfmt(metrics.mean_squared_error(y_test, predictions))
init_r2 = numfmt(metrics.r2_score(y_test, predictions))
print('the initial MSE currently stands at: {}; '.format(init_mse), 'the initial R-squared stands at: {}'.format(init_r2))

# preparing Gradient Descent

old_theta = lr.coef_  # capturing parameters from linear model
initial_params = np.ones(old_theta.shape)  # generating initial parameters using the shape of existing ones
new_theta, cost_set = gradient_descent(initial_params, x_test, y_test)

print(' old thetas are: ', [numfmt(i) for i in old_theta], '\n', 'new thetas are: ', [numfmt(i) for i in new_theta])
plt.plot(range(len(cost_set)), cost_set)
plt.show()

# applying new parameters
lr.coef_ = new_theta
predictions = lr.predict(x_test)

# calculating new metrics
new_r2 = numfmt(metrics.r2_score(y_test, predictions))
new_mse = numfmt(metrics.mean_squared_error(y_test, predictions))

print('the new MSE currently stands at: {}; '.format(new_mse), 'the R-squared stands at: %s' % new_r2)

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


kf_mse = cross_validation.cross_val_score(lr, sm.add_constant(normalize(regressors)), regressand, scoring='mean_squared_error', cv=kf_gen)
kf_r2 = cross_validation.cross_val_score(lr, sm.add_constant(normalize(regressors)), regressand, scoring='r2', cv=kf_gen)
print('the average MSE from k-fold validation: {}; '.format(numfmt(np.mean(abs(kf_mse)))),
      'the average R-squared stands at: {}'.format(numfmt(np.mean(kf_r2))))

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

x_train, x_test, y_train, y_test = cross_validation.train_test_split(data[cat_cols], data['origin']\
                                                                     , test_size=.2, random_state=2)

unique_origins = sorted(data['origin'].unique())
testing_probs = pd.DataFrame(columns=unique_origins)

models = dict()  # dict to contain different classifiers with each to produce one binomial output.

for i in unique_origins:
    classifier = linear_model.LogisticRegression()
    x = x_train
    y = y_train == i
    classifier.fit(x, y)
    models[i] = classifier
    testing_probs[i] = models[i].predict_proba(x_test)[:, 1]  # only capturing probability of positive result

test_set = x_test.join(y_test).reset_index(drop=True)
predictions = testing_probs.idxmax(axis=1)
test_set['predicted_origin'] = predictions
correct = test_set['origin'] == test_set['predicted_origin']


# applying trained classifiers to full data

probs = pd.DataFrame(columns=unique_origins)

for i in unique_origins:
    probs[i] = models[i].predict_proba(data[cat_cols])[:, 1]

data['predicted_origin'] = probs.idxmax(axis=1)

print('classier accuracy on testing stands at:', numfmt(len(test_set[correct]) / len(test_set)))
print('classier accuracy on whole data stands at:', numfmt(len(data[data['origin'] == data['predicted_origin']]) / len(data)))


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

