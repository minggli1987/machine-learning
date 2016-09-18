from sklearn import metrics, cross_validation, naive_bayes, preprocessing, pipeline, linear_model, tree, neural_network
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from sklearn.externals.six import StringIO
import statsmodels.api as sm
import pydotplus
warnings.filterwarnings('ignore')

data = load_iris()


def visualize():

    # Code source: Gaël Varoquaux
    # Modified for documentation by Jaques Grobler
    # License: BSD 3 clause

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn import datasets
    from sklearn.decomposition import PCA

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    Y = iris.target

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    # plt.figure(2, figsize=(8, 6))
    # plt.clf()
    #
    # # Plot the training points
    # plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    # plt.xlabel('Sepal length')
    # plt.ylabel('Sepal width')
    #
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())

    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(iris.data)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,
               cmap=plt.cm.Paired)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()


target_dict = {'species': {k: v for k, v in enumerate(data['target_names'])}}

df = pd.DataFrame(data['data'], columns=data['feature_names'], dtype=float) \
    .join(
    pd.DataFrame(data['target'], columns=['species'], dtype=int)).replace(target_dict)

df['species'] = pd.Categorical.from_array(df['species']).codes
df['species'] = df['species'].astype('category')

regressors = df.select_dtypes(include=['float'])
regressand = df.select_dtypes(include=['category'])


reg = linear_model.LogisticRegression()
reg = naive_bayes.GaussianNB()
reg = tree.DecisionTreeClassifier(max_depth=3, max_leaf_nodes=20, min_samples_leaf=30, random_state=2)
reg.fit(X=regressors, y=regressand)


kf_gen = cross_validation.KFold(df.shape[0], n_folds=5, shuffle=False, random_state=2)

prediction = cross_validation.cross_val_predict(reg, X=regressors, y=regressand, cv=kf_gen, n_jobs=-1)
df['pred'] = prediction
accuracy = cross_validation.cross_val_score(reg, regressors, regressand, scoring='accuracy', cv=kf_gen, n_jobs=-1)
print(np.mean(accuracy))

# with pd.ExcelWriter('iris_pred.xlsx') as writer:
#     df.to_excel(writer, sheet_name='output', index=False)

dot_data = StringIO()
tree.export_graphviz(reg, feature_names=regressors.columns, class_names=target_dict['species'], filled=True, \
                     rounded=True, special_characters=True, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

# neutral network applying on iris dataset from dataquest

class NNet3:

    def __init__(self, learning_rate=0.5, maxepochs=1e4, convergence_thres=1e-5, hidden_layer=4):
        self.learning_rate = learning_rate
        self.maxepochs = int(maxepochs)
        self.convergence_thres = 1e-5
        self.hidden_layer = int(hidden_layer)

    def _multiplecost(self, X, y):
        # feed through network
        l1, l2 = self._feedforward(X)
        # compute error
        inner = y * np.log(l2) + (1 - y) * np.log(1 - l2)
        # negative of average error
        return -np.mean(inner)

    def _sigmoid_activation(self, X, theta):
        X = np.asarray(X)
        theta = np.asarray(theta)
        return 1 / (1 + np.exp(-np.dot(theta.T, X)))  # logistic sigmoid

    def _feedforward(self, X):
        # feedforward to the first layer
        l1 = self._sigmoid_activation(X.T, self.theta0).T
        # add a column of ones for bias term
        l1 = np.column_stack([np.ones(l1.shape[0]), l1])
        # activation units are then inputted to the output layer
        l2 = self._sigmoid_activation(l1.T, self.theta1)
        return l1, l2

    def predict(self, X):
        _, y = self._feedforward(X)
        return y

    def learn(self, X, y):

        nobs, ncols = X.shape

        # initial thetas

        self.theta0 = np.random.normal(0, 0.01, size=(ncols, self.hidden_layer))
        self.theta1 = np.random.normal(0, 0.01, size=(self.hidden_layer + 1, 1))

        self.costs = []
        cost = self._multiplecost(X, y)
        self.costs.append(cost)
        costprev = cost + self.convergence_thres + 1  # set an inital costprev to past while loop
        counter = 0  # intialize a counter

        # Loop through until convergence
        for counter in range(self.maxepochs):
            # feedforward through network
            l1, l2 = self._feedforward(X)

            # Start Backpropagation
            # Compute gradients
            l2_delta = (y - l2) * l2 * (1 - l2)
            l1_delta = l2_delta.T.dot(self.theta1.T) * l1 * (1 - l1)

            # Update parameters by averaging gradients and multiplying by the learning rate
            self.theta1 += l1.T.dot(l2_delta.T) / nobs * self.learning_rate
            self.theta0 += X.T.dot(l1_delta)[:, 1:] / nobs * self.learning_rate

            # Store costs and check for convergence
            counter += 1  # Count
            costprev = cost  # Store prev cost
            cost = self._multiplecost(X, y)  # get next cost
            self.costs.append(cost)
            if np.abs(costprev - cost) < self.convergence_thres and counter > 500:
                break

    def show_cost(self):
        # Plot costs
        plt.plot(self.costs)
        plt.title("Convergence of the Cost Function")
        plt.ylabel("J($\Theta$)")
        plt.xlabel("Iteration")
        plt.show()

# Set a learning rate
learning_rate = 0.5
# Maximum number of iterations for gradient descent
maxepochs = 10000
# Costs convergence threshold, ie. (prevcost - cost) > convergence_thres
convergence_thres = 0.00001
# Number of hidden units
hidden_units = 4

# Initialize model
model = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,
              convergence_thres=convergence_thres, hidden_layer=hidden_units)
# Train model

X = np.column_stack([np.ones(regressors.shape[0]), np.array(regressors)])
y = np.array((regressand['species'] == 1).values.astype(int))
print(y)

x_train, x_test, y_train, y_test = \
    cross_validation.train_test_split(X, y, test_size=.3)

model.learn(x_train, y_train)

yhat = model.predict(x_test)[0]
auc = metrics.roc_auc_score(y_test, yhat)
print(auc)


def sigmoid_activation(X, theta):
    X = np.asarray(X)
    theta = np.asarray(theta)
    return 1 / (1 + np.exp(-np.dot(theta.T, X)))  # logistic sigmoid


l1 = sigmoid_activation(x_test.T, model.theta0).T
# add a column of ones for bias term
l1 = np.column_stack([np.ones(l1.shape[0]), l1])
# activation units are then inputted to the output layer
l2 = sigmoid_activation(l1.T, model.theta1)
print(model.theta1.T.shape, l1.T.shape, l2.shape)
