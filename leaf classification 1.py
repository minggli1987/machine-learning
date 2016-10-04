
# coding: utf-8

# In[45]:

__author__ = 'Ming Li'
# This application forms a submission from Ming in regards to leaf classification challenge on Kaggle community
# import tensorflow as tf
# from tensorflow.contrib import learn
from GradientDescent import GradientDescent
from sklearn import metrics, cross_validation, naive_bayes, preprocessing, pipeline, linear_model, tree, decomposition, ensemble
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
import pydotplus
warnings.filterwarnings('ignore')


# In[46]:

# set display right
pd.set_option('display.width', 4000)
pd.set_option('max_colwidth', 4000)
pd.set_option('max_rows', 100)
pd.set_option('max_columns', 200)
pd.set_option('float_format', '%.9f')


# In[47]:

test = pd.read_csv('data/leaf/test.csv')
train = pd.read_csv('data/leaf/train.csv')


# In[48]:

regressors = train.select_dtypes(exclude=(np.int, np.int64, np.object)).copy()
regressand = train.select_dtypes(exclude=(np.int, np.int64, np.float)).copy()


# # codifying types of species

# In[49]:

regressand['species_id'] = pd.Categorical.from_array(regressand['species']).codes
mapping = regressand[['species_id','species']].set_index('species_id').to_dict()['species']
regressand.drop('species', axis=1, inplace=True)


# # model generalization

# In[50]:

kf_generator = cross_validation.KFold(train.shape[0], n_folds=5, shuffle=True, random_state=1)


# # feature scaling

# In[51]:

regressors_std = regressors.apply(preprocessing.scale, with_mean=False, with_std=True, axis=0)
# using standard deviation as denominator


# # Logistic Regression

# In[52]:

regressors_std = np.column_stack((np.ones(regressors.shape[0]), regressors_std))  # add constant 1


# In[53]:

regressors = np.column_stack((np.ones(regressors.shape[0]), regressors))  # add constant 1


# In[54]:

x_train, x_test, y_train, y_test = cross_validation.train_test_split(regressors_std, regressand, test_size=.2, random_state=1)

reg = linear_model.LogisticRegression(fit_intercept=False)  # regressors already contains manually added intercept
reg.fit(x_train, y_train)

prediction = reg.predict(x_test)
print(metrics.accuracy_score(y_test, prediction))

# scores = cross_validation.cross_val_score(reg, regressors_std, regressand, scoring='accuracy', cv=kf_generator)
# print(np.mean(scores))

# # gradient descent optimisation algorithm

old_theta = reg.coef_
gd = GradientDescent(alpha=.1, max_epochs=5000, conv_thres=.0000001, display=True)
gd.fit(x_train, y_train, reg)
new_theta, costs = gd.optimise()
print(new_theta.shape)
plt.plot(range(len(costs[0])), costs[0])
plt.show()
# applying new parameters after optimisation
reg.coef_ = new_theta

prediction = reg.predict(x_test)
print(metrics.accuracy_score(y_test, prediction))
# scores = cross_validation.cross_val_score(reg, regressors_std, regressand, scoring='accuracy', cv=kf_generator)
# print(np.mean(scores))

# # apply trained model

test['species'] = np.nan
test = test[train.columns]

combined = pd.concat([test, train])

combined.sort_values('id', inplace=True)

regressors = combined.select_dtypes(exclude=(np.int, np.int64, np.object)).copy()
regressors_std = regressors.apply(preprocessing.scale, axis=0)  # using standard deviation as denominator
regressors_std = np.column_stack((np.ones(regressors_std.shape[0]), regressors_std))  # add constant 1

combined['species_id'] = reg.predict(regressors_std)

combined['species_predicted'] = combined['species_id'].map(mapping)

result = combined.select_dtypes(include=(np.int, np.int64, np.object)).copy()

probs = reg.predict_proba(regressors_std)

probs_df = pd.DataFrame(probs, columns=mapping.values(), index=combined['id'])

submission = probs_df.loc[test['id']]

submission.to_csv('data/leaf/submission.csv', encoding='utf-8', header=True)
