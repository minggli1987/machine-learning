from sklearn import metrics, cross_validation, naive_bayes, preprocessing, pipeline, linear_model, tree, decomposition, ensemble
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import sys
from visual import copy_pics_into_folders, delete_folders
from GradientDescent import GradientDescent

warnings.filterwarnings('ignore')

# coding: utf-8

__author__ = 'Ming Li'
# This application forms a submission from Ming in regards to leaf classification challenge on Kaggle community

# set display right
pd.set_option('display.width', 4000)
pd.set_option('max_colwidth', 4000)
pd.set_option('max_rows', 100)
pd.set_option('max_columns', 200)
pd.set_option('float_format', '%.4f')


# load raw data

test = pd.read_csv('leaf/test.csv')
train = pd.read_csv('leaf/train.csv')


# picking useful data points

regressors = train.select_dtypes(exclude=(np.int8, np.int64, np.object)).copy()
regressand = train.select_dtypes(exclude=(np.int8, np.int64, np.float)).copy()

# codifying types of species

regressand['species_id'] = pd.Categorical.from_array(regressand['species']).codes
mapping = regressand[['species_id','species']].set_index('species_id').to_dict()['species']
regressand.drop('species', axis=1, inplace=True)
regressand = np.ravel(regressand)


# model generalization

kf_generator = cross_validation.StratifiedKFold(regressand, n_folds=5, shuffle=False, random_state=1)

# feature scaling using standard deviation as denominator

regressors_std = regressors.apply(preprocessing.scale, with_mean=False, with_std=True, axis=0)

# Logistic regression

regressors_std = np.column_stack((np.ones(regressors.shape[0]), regressors_std))  # add constant
regressors = np.column_stack((np.ones(regressors.shape[0]), regressors))  # add constant

# hold out

x_train, x_test, y_train, y_test = cross_validation.\
    train_test_split(regressors_std, regressand, test_size=.2, random_state=1)

# fit training set

reg = linear_model.LogisticRegression(fit_intercept=False)  # regressors already contains manually added intercept
reg.fit(x_train, y_train)


print('Using given features by Kaggle, Logistic Regression model accuracy is:', flush=True, end=' ')

avg_scores = cross_validation.cross_val_score(reg, regressors_std, regressand, scoring='accuracy', cv=kf_generator)
print('{0:.2f}%'.format(100 * np.mean(avg_scores)), flush=True, end='\n')


def grad_student_descent():

    # Gradient Descent optimisation algorithm

    old_theta = np.ones(reg.coef_.shape)
    gd = GradientDescent(alpha=.1, max_epochs=5000, conv_thres=.0000001, display=False)
    gd.fit(x_train, y_train, reg)
    gd.optimise()
    new_theta, costs = gd.thetas, gd.costs

    # applying new parameters after optimisation

    reg.coef_ = new_theta

    prediction = reg.predict(x_test)
    print(metrics.accuracy_score(y_test, prediction))
    scores = cross_validation.cross_val_score(reg, regressors_std, regressand, scoring='accuracy', cv=kf_generator)
    print(np.mean(scores))


# combine train and test

test['species'] = np.nan  # adding species to testing set
test = test[train.columns]  # only keep columns in training

combined = pd.concat(objs=[test, train], axis=0).sort_values('id', ascending=True)
regressors = combined.select_dtypes(exclude=(np.int8, np.int64, np.object)).copy()
regressors_std = regressors.apply(preprocessing.scale, axis=0)  # using standard deviation as denominator
regressors_std = np.column_stack((np.ones(regressors_std.shape[0]), regressors_std))  # add constant


# moving classified pictures into folders

combined['species_id'] = reg.predict(regressors_std)  # making predictions
combined['species_predicted'] = combined['species_id'].map(mapping)
result = combined.select_dtypes(include=(np.int8, np.int64, np.object)).copy()
table = result[['species_predicted', 'id']].sort_values(by=['species_predicted', 'id'], ascending=True)\
    .set_index('species_predicted')

copy_pics_into_folders(table)

key = input('continue to produce submission?\n')

if str(key).lower() != 'yes':
    delete_folders(table)
    sys.exit()

else:
    # produce submission.csv as requested by kaggle
    print('producing submission...', flush=True, end='')
    probs = reg.predict_proba(regressors_std)
    probs_df = pd.DataFrame(probs, columns=mapping.values(), index=combined['id'])
    submission = probs_df.loc[test['id']]
    submission.to_csv('submission.csv', encoding='utf-8', header=True)
    delete_folders(table)
    print('done', flush=True)
    sys.exit()