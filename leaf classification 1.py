
# coding: utf-8

# In[45]:

__author__ = 'Ming Li'
# This application forms a submission from Ming in regards to leaf classification challenge on Kaggle community
import tensorflow as tf
from tensorflow.contrib import learn
from minglib import gradient_descent
from sklearn import metrics, cross_validation, naive_bayes, preprocessing, pipeline, linear_model, tree, decomposition, ensemble
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from sklearn.externals.six import StringIO
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from minglib import gradient_descent
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

regressors = train.select_dtypes(exclude=(np.int, np.object)).copy()
regressand = train.select_dtypes(exclude=(np.int, np.float)).copy()


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


# In[55]:

reg = linear_model.LogisticRegression(fit_intercept=False)  # regressors already contains manually added intercept


# In[56]:

reg.fit(x_train, y_train)


# In[57]:

prediction = reg.predict(x_test)


# In[58]:

reg.coef_.shape # 99 one-vs-rest logistic regression coefficients x 192 features


# In[59]:

metrics.accuracy_score(y_test, prediction)


# In[60]:

scores = cross_validation.cross_val_score(reg, regressors_std, regressand, scoring='accuracy', cv=kf_generator)
np.mean(scores)


# # gradient descent optimisation algorithm

# In[61]:

old_theta = reg.coef_


# In[62]:

old_theta.shape


# In[63]:

new_theta, costs = gradient_descent(old_theta, x_train, y_train, reg, alpha=.1)


# In[64]:

reg.coef_ = new_theta


# In[65]:

prediction = reg.predict(x_test)


# In[66]:

metrics.accuracy_score(y_test, prediction)


# In[67]:

scores = cross_validation.cross_val_score(reg, regressors_std, regressand, scoring='accuracy', cv=kf_generator)
np.mean(scores)


# # apply trained model

# In[272]:

test['species'] = np.nan
test = test[train.columns]


# In[273]:

combined = pd.concat([test,train])


# In[274]:

combined.sort_values('id', inplace=True)


# In[275]:

regressors = combined.select_dtypes(exclude=('int', 'object')).copy()
regressors_std = regressors.apply(preprocessing.scale, axis=0)  # using standard deviation as denominator
regressors_std = np.column_stack((np.ones(regressors_std.shape[0]), regressors_std))  # add constant 1


# In[276]:

combined['species_id'] = reg.predict(regressors_std)


# In[277]:

combined['species_predicted'] = combined['species_id'].map(mapping)


# In[278]:

result = combined.select_dtypes(include=('int','object')).copy()


# In[279]:

probs = reg.predict_proba(regressors_std)


# In[280]:

probs_df = pd.DataFrame(probs, columns=mapping.values(), index=combined['id'])


# In[281]:

submission = probs_df.loc[test['id']]


# In[282]:

submission.to_csv('data/leaf/submission.csv', encoding='utf-8', header=True)


# # Bayesian?

# In[77]:

x_train, x_test, y_train, y_test = cross_validation.train_test_split(regressors, regressand, test_size=.2, random_state=1)


# In[78]:

clf = naive_bayes.GaussianNB()


# In[79]:

clf.fit(x_train, y_train)


# In[80]:

prediction = clf.predict(x_test)


# In[81]:

metrics.accuracy_score(y_test, prediction)


# In[82]:

scores = cross_validation.cross_val_score(clf, regressors, regressand, scoring='accuracy', cv=kf_generator)
np.mean(scores)


# # Tree

# In[83]:

clf = ensemble.RandomForestClassifier(max_depth=100, max_leaf_nodes=500, min_samples_leaf=3, random_state=1)


# In[84]:

clf.fit(x_train, y_train)


# In[85]:

prediction = clf.predict(x_test)


# In[86]:

metrics.accuracy_score(y_test, prediction)


# In[87]:

scores = cross_validation.cross_val_score(clf, regressors, regressand, scoring='accuracy', cv=kf_generator)
np.mean(scores)


# # Neural Network

# In[3]:

features = regressors.astype(float)
target = np.array(regressand).astype(int)


# In[4]:

x_train, x_test, y_train, y_test = cross_validation.train_test_split(features, target, test_size=.2, random_state=1)


# In[5]:

def main(unused_argv):

#     iris = learn.datasets.load_dataset('iris')
#     x_train, x_test, y_train, y_test = cross_validation.train_test_split(
#         iris.data, iris.target, test_size=0.2, random_state=42)

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    feature_columns = learn.infer_real_valued_columns_from_input(x_train)
    classifier = learn.DNNClassifier(
        feature_columns=feature_columns, hidden_units=[20, 20, 20, 20], n_classes=99)

    # Fit and predict.
    classifier.fit(x_train, y_train, steps=200)
    predictions = list(classifier.predict(x_test, as_iterable=True))
    score = metrics.accuracy_score(y_test, predictions)
    print('Accuracy: {:.4f}'.format(score))
    # print(y_test, predictions)


# In[6]:

if __name__ == '__main__':
  output = tf.app.run()


# In[7]:

predictions


# In[8]:




# In[8]:




# In[ ]:




# In[ ]:




# In[ ]:



