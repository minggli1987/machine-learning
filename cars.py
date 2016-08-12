
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import preprocessing, cluster, metrics, cross_validation, linear_model


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

lr = linear_model.LinearRegression(fit_intercept=False).fit(normalized_regressors, regressand)

data['predicted_acceleration'] = lr.predict(normalized_regressors)


# fig, ax = plt.subplots()
# ax.scatter(data['horsepower'], data['acceleration'], color='b')
# ax.scatter(data['horsepower'], data['predicted_acceleration'], color='r')
# ax.set_xlabel('horsepower')
# ax.set_ylabel('acceleration')
# plt.show()


# calculating MSE for linear model
init_mse = metrics.mean_squared_error(regressand, data['predicted_acceleration'])
init_r2 = metrics.r2_score(regressand, data['predicted_acceleration'])
print('the initial MSE currently stands at: ', init_mse, '\n', ' and the R-squared stands at: ', init_r2)

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


cost_function(old_theta, normalized_regressors, regressand)

initial_params = np.ones(old_theta.shape)

new_theta, cost_set = gradient_descent(initial_params, normalized_regressors, regressand)

print('old thetas are:  ', old_theta, '\n', 'new thetas are: ', new_theta)

plt.plot(range(len(cost_set)), cost_set)
plt.show()

# applying new parameters

lr.coef_ = new_theta
data['predicted_acceleration'] = lr.predict(normalized_regressors)

residuals = - data['acceleration'] + data['predicted_acceleration']


new_r2 = metrics.r2_score(regressand, data['predicted_acceleration'])
new_mse = metrics.mean_squared_error(regressand, data['predicted_acceleration'])
print('the new MSE currently stands at: ', new_mse, '\n', ' and the R-squared stands at: ', new_r2)


# multi-class Classifier on Origin

# transforming
dummy_cylinders = pd.get_dummies(data['cylinders'], prefix='cyl')
dummy_years = pd.get_dummies(data['year'], prefix='y')
dummies = pd.concat([dummy_cylinders, dummy_years], axis=1)
data = data.join(dummies)

cat_cols = [col for col in data.columns if col.startswith('y_') or col.startswith('cyl_')]

x_train, x_test, y_train, y_test = cross_validation.train_test_split(data[cat_cols], data['origin'] \
                                                                     , test_size=.2, random_state=1)

models = dict()
unique_origins = data['origin'].unique()
testing_probs = pd.DataFrame(columns=unique_origins)


for i in unique_origins:
    classifier = linear_model.LogisticRegression()
    x = x_train
    y = y_train == i
    classifier.fit(x, y)
    models[i] = classifier
    testing_probs[i] = models[i].predict_proba(x_test)[:, 1]

test_set = x_test.join(y_test).reset_index(drop=True)

predictions = testing_probs.idxmax(axis=1)

test_set['predicted_origin'] = predictions

correct = test_set['origin'] == test_set['predicted_origin']

print(len(test_set[correct]) / len(test_set))

probs = pd.DataFrame(columns=unique_origins)

for i in unique_origins:
    probs[i] = models[i].predict_proba(data[cat_cols])[:, 1]

data['predicted_origin'] = probs.idxmax(axis=1)

print(len(data[data['origin'] == data['predicted_origin']]) / len(data))


# Clustering Cars

data = data[[i for i in data.columns if (i not in dummies.columns) and (i not in ['cylinders', 'year'])]]

data['hw_ratio'] = data['horsepower'] / data['weight']

plt.scatter(data['horsepower'], data['weight'])




clustering = cluster.KMeans(n_clusters=3, random_state=1)


# In[191]:

clustering.fit(data[['horsepower','weight']])


# In[192]:

data['cluster'] = clustering.labels_


# In[193]:

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


# In[194]:

for n in range(3):
    clustered_data = data[data['cluster'] == n]
    plt.scatter(clustered_data['horsepower'], clustered_data['weight'], color=colors[n])


# In[195]:

legend = {1:'North America', 2: 'Europe', 3:'Asia'}




for n in range(1,4,1):
    clustered_data = data[data['origin'] == n]
    plt.scatter(clustered_data['horsepower'], clustered_data['weight'], color=colors[n], label=legend[n])
plt.legend(loc=2)


# In[ ]:




# In[ ]:



