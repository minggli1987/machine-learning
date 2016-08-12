

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import preprocessing, cluster, metrics, cross_validation, linear_model.LogisticRegression, linear_model.LinearRegression


data = pd.read_csv('data/auto-mpg.data-original.txt', header=None, delim_whitespace=True)

cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin', 'name']

data.columns = cols
data = data.dropna().reset_index(drop=True)
data['year'] = (1900 + data['year']).astype(int)
data.ix[:, :5] = data.ix[:, :5].astype(int)
data['origin'] = data['origin'].astype(int)

# defining predictive variables
regressors = data[['mpg', 'displacement', 'horsepower', 'weight']]

normalized_regressors = sm.add_constant(preprocessing.scale(regressors))

regressand = data['acceleration']

lr = LinearRegression(fit_intercept=False)

lr.fit(normalized_regressors, regressand)

data['predicted_acceleration'] = lr.predict(normalized_regressors)


fig, ax = plt.subplots()
ax.scatter(data['horsepower'], data['acceleration'], color='b')
ax.scatter(data['horsepower'], data['predicted_acceleration'], color='r')
ax.set_xlabel('horsepower')
ax.set_ylabel('acceleration')
plt.show()

# In[211]:

metrics.mean_squared_error(regressand, data['predicted_acceleration'])


# In[212]:

theta = lr.coef_


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
    #max number of iterations
    max_epochs = 10000
    #initaiting a count number so once reaching max iterations will termiante
    count = 0
    #convergence threshold
    conv_thres = 0.00000001
    #initail cost
    cost = cost_function(params, x, y)
    prev_cost = cost + 10
    costs = [cost]
    thetas = [params]
    
    #beginning gradient_descent iterations
    
    while (np.abs(prev_cost - cost) > conv_thres) and (count <= max_epochs):
        prev_cost = cost
        update = np.ones(len(params))
        #simutaneously update all thetas
        for j in range(len(params)):
            update[j] = alpha * partial_derivative_cost(params, j, x, y)

        params -= update
        
        thetas.append(params)
        
        cost = cost_function(params, x, y)
        
        costs.append(cost)
        count += 1
        
    return params, costs


cost_function(theta, normalized_regressors, regressand)


# In[220]:

new_theta, cost_set = gradient_descent([10,10,10,10,10], normalized_regressors, regressand)


# In[221]:

new_theta


# In[222]:

cost_function(new_theta, normalized_regressors, regressand)


# In[223]:

plt.plot(range(len(cost_set)), cost_set)


# In[224]:

lr.coef_ = new_theta


# In[225]:

residuals = - data['acceleration'] + data['predicted_acceleration']


# In[226]:

residuals.hist()


# In[227]:

r2 = r2_score(data['acceleration'], data['predicted_acceleration'])


# In[228]:

mean_squared_error(regressand, data['predicted_acceleration'])


# # Multiclass Classifier on Origin

# In[170]:

dummy_cylinders = pd.get_dummies(data['cylinders'], prefix='cyl')


# In[171]:

dummy_years = pd.get_dummies(data['year'], prefix='y')


# In[172]:

dummies = pd.concat([dummy_cylinders, dummy_years], axis=1)


# In[173]:

data = data.join(dummies)


# In[174]:

categorical_cols = [col for col in data.columns if col.startswith('y_') or col.startswith('cyl_')]


# In[175]:

x_train, x_test, y_train, y_test = train_test_split(data[categorical_cols],data['origin'],test_size=.2, random_state=1)


# In[176]:

models = dict()
unique_origins = data['origin'].unique()
testing_probs = pd.DataFrame(columns = unique_origins)


# In[177]:

for i in unique_origins:
    classifier = LogisticRegression()
    x = x_train
    y = y_train == i
    classifier.fit(x, y)
    models[i] = classifier
    testing_probs[i] = models[i].predict_proba(x_test)[:,1]


# In[178]:

test_set = x_test.join(y_test).reset_index(drop=True)


# In[179]:

predictions = testing_probs.idxmax(axis=1)


# In[180]:

test_set['predicted_origin'] = predictions


# In[181]:

correct = test_set['origin'] == test_set['predicted_origin']


# In[182]:

len(test_set[correct]) / len(test_set)


# In[183]:

probs = pd.DataFrame(columns = unique_origins)


# In[184]:

for i in unique_origins:
    probs[i] = models[i].predict_proba(data[categorical_cols])[:,1]


# In[185]:

data['predicted_origin'] = probs.idxmax(axis=1)


# In[186]:

len(data[data['origin'] == data['predicted_origin']]) / len(data)


# # Clustering Cars

# In[187]:

data = data[[i for i in data.columns if (i not in dummies.columns) and (i not in ['cylinders','year'])]]


# In[188]:

data['hw_ratio'] = data['horsepower'] / data['weight']


# In[189]:

plt.scatter(data['horsepower'], data['weight'])


# In[190]:

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


# In[196]:

for n in range(1,4,1):
    clustered_data = data[data['origin'] == n]
    plt.scatter(clustered_data['horsepower'], clustered_data['weight'], color=colors[n], label=legend[n])
plt.legend(loc=2)


# In[ ]:




# In[ ]:



