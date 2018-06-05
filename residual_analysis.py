import pandas as pd

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

dataset = load_boston(return_X_y=False)

X = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
y = pd.DataFrame(dataset['target'], columns=['PRICE'])

values = MinMaxScaler().fit_transform(X)
X_scaled = pd.DataFrame(values, columns=X.columns, index=X.index)

lr = LinearRegression()
lr.fit(X, y)
pred = lr.predict(X_scaled)

residuals = y - pred
y = y.rename(columns={'PRICE': 'residuals'})

for col in X:
    sns.lmplot(col, 'residuals', X_scaled.join(y), fit_reg=False, scatter=True)

plt.show()
