import pandas as pd
import numpy as np
from sklearn import model_selection

__author__ = 'Ming Li'


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
genderclassmodel = pd.read_csv('data/genderclassmodel.csv')
gendermodel = pd.read_csv('data/gendermodel.csv')


def transform(x):

    assert isinstance(x, pd.DataFrame), 'require a dataframe'

    mapping = {
        'Sex': {'male': 1, 'female': 0},
        'Embarked': {None: 0, 'S': 1, 'C': 2, 'Q': 3}
    }
    data = x.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    data.replace(mapping, inplace=True)
    data[['Sex', 'Embarked']] = data[['Sex', 'Embarked']].astype(np.int)
    data['Survived'] = data['Survived'].astype('category')

    return data

data = transform(train)

regressors = data.select_dtypes(exclude=['category'])
regressand = data.select_dtypes(include=['category'])

# splitting and hold out

kf_gen = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)
x_train, x_test, y_train, y_test = \
    model_selection.train_test_split(regressors, regressors, test_size=.2, random_state=1)
