
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import preprocessing, cluster, metrics, cross_validation, linear_model


class NumberFormatted(object):
    def __init__(self, n):
        self.value = float(n)

    def converted(self):
        return '{0:.2f}'.format(self.value)

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin', 'name']
data = pd.read_csv('data/auto-mpg.data-original.txt', header=None, delim_whitespace=True, names=cols)

data = data.dropna().reset_index(drop=True)
data['year'] = (1900 + data['year']).astype(int)
data.ix[:, :5] = data.ix[:, :5].astype(int)
data['origin'] = data['origin'].astype(int)

# defining predictive variables
regressors = data[['mpg', 'displacement', 'horsepower', 'weight']]

normalized_regressors = sm.add_constant(preprocessing.scale(regressors))

regressand = data['acceleration']

x_train, x_test, y_train, y_test = cross_validation.train_test_split(normalized_regressors, regressand\
                                                                     , test_size=.2, random_state=1)

lr = linear_model.LinearRegression(fit_intercept=False).fit(x_train, y_train)

predictions = lr.predict(x_test)

# fig, ax = plt.subplots()
# ax.scatter(data['horsepower'], data['acceleration'], color='b')
# ax.scatter(data['horsepower'], data['predicted_acceleration'], color='r')
# ax.set_xlabel('horsepower')
# ax.set_ylabel('acceleration')
# plt.show()

# calculating MSE for linear model
init_mse = metrics.mean_squared_error(y_test, predictions)
init_r2 = metrics.r2_score(y_test, predictions)
print('the initial MSE currently stands at: ', NumberFormatted(init_mse).converted(), '\n', ' and the R-squared stands at: ', NumberFormatted(init_r2).converted())

old_theta = lr.coef_  # capturing parameters from linear model

# cost function, partial_derivative and gradient descent algorithm


def cost_function(params, x, y):
    params = np.array(params)
    x = np.array(x)
    y = np.array(y)
    J = 0
    m = len(x)
    for i in range(m):
        h = np.sum(params.T * x[i])
        diff = (h - y[i])**2
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

    print('beginning gradient decent algorithm')
    
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


cost_function(old_theta, x_test, y_test)

initial_params = np.zeros(old_theta.shape)

new_theta, cost_set = gradient_descent(initial_params, x_test, y_test)

print('old thetas are:  ', old_theta, '\n', 'new thetas are: ', new_theta)

plt.plot(range(len(cost_set)), cost_set)
plt.show()

# applying new parameters

lr.coef_ = new_theta
predictions = lr.predict(x_test)

# displaying new R-squared and MSE
new_r2 = metrics.r2_score(y_test, predictions)
new_mse = metrics.mean_squared_error(y_test, predictions)
print('the new MSE currently stands at: ', NumberFormatted(new_mse).converted(), '\n', ' and the R-squared stands at: ', NumberFormatted(new_r2).converted())


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

print('classier accuracy on testing stands at:', NumberFormatted(len(test_set[correct]) / len(test_set)).converted())
print('classier accuracy on whole data stands at:', NumberFormatted(len(data[data['origin'] == data['predicted_origin']]) / len(data)).converted())


# Clustering Cars

data = data[[i for i in data.columns if (i not in dummies.columns) and (i not in ['cylinders', 'year'])]]

plt.scatter(data['horsepower'], data['weight'])
plt.show()

clustering = cluster.KMeans(n_clusters=3, random_state=1).fit(data[['horsepower', 'weight']])

data['cluster'] = clustering.labels_

scatter_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

for n in range(len(set(clustering.labels_))):
    clustered_data = data[data['cluster'] == n]
    plt.scatter(clustered_data['horsepower'], clustered_data['weight'], color=scatter_colors[n])
plt.show()

legend = {1: 'North America', 2: 'Europe', 3: 'Asia'}

for n in range(1, len(set(clustering.labels_)) + 1, 1):
    clustered_data = data[data['origin'] == n]
    plt.scatter(clustered_data['horsepower'], clustered_data['weight'], color=scatter_colors[n], label=legend[n])
plt.legend(loc=2)
plt.show()

