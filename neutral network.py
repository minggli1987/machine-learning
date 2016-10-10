import numpy as np
from sklearn import datasets, cross_validation, metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# neutral network applying on iris dataset from dataquest

__author__ = 'Ming Li'


class NeuralNet:

    def __init__(self, alpha=0.5, maxepochs=1e4, conv_thres=1e-5, hidden_layer=4):
        self._alpha = alpha
        self._maxepochs = maxepochs
        self._conv_thres = conv_thres
        self._hidden_layer = hidden_layer

    def __multiple_cost_func__(self, X, y):

        # feed through network
        l1, l2 = self.__feed_forward__(X)
        # compute error
        inner = y * np.log(l2) + (1 - y) * np.log(1 - l2)
        # negative of average error
        return -np.mean(inner)

    def __sigmoid_activation__(self, X, theta):

        X = np.array(X)
        theta = np.array(theta)

        return 1 / (1 + np.exp(-np.dot(theta.T, X)))  # logistic sigmoid function

    def __feed_forward__(self, X):
        # feedforward to the initial layer
        l1 = self.__sigmoid_activation__(X.T, self.theta0).T
        # add a column of ones for bias term
        l1 = np.column_stack([np.ones(l1.shape[0]), l1])
        # activation units are then inputted to the output layer
        l2 = self.__sigmoid_activation__(l1.T, self.theta1)
        return l1, l2

    def predict(self, X):
        _, y = self.__feed_forward__(X)
        return y

    def learn(self, X, y):

        n_samples, n_feature = X.shape

        # initial thetas

        self.theta0 = np.random.normal(0, 0.01, size=(n_feature, self._hidden_layer))
        self.theta1 = np.random.normal(0, 0.01, size=(self._hidden_layer + 1, 1))  # plus one bias term
        print(self.theta0.shape, self.theta1.shape)

        self.cost_set = []
        cost = self.__multiple_cost_func__(X, y)
        self.cost_set.append(cost)
        
        costprev = cost + self._conv_thres + 1  # set an inital costprev to past while loop
        count = 0  # setting a counter

        # Loop through until convergence

        while np.abs(costprev - cost) < self._conv_thres and count < self._maxepochs:

            # feed forward through network

            l1, l2 = self.__feed_forward__(X)

            # Start Backpropagation

            # Compute gradients
            l2_delta = (y - l2) * l2 * (1 - l2)
            l1_delta = l2_delta.T.dot(self.theta1.T) * l1 * (1 - l1)

            # Update parameters by averaging gradients and multiplying by the learning rate
            self.theta0 += X.T.dot(l1_delta)[:, 1:] / n_samples * self._alpha
            self.theta1 += l1.T.dot(l2_delta.T) / n_samples * self._alpha

            # Store costs and check for convergence
            count += 1  # Count
            costprev = cost  # Store prev cost
            cost = self.__multiple_cost_func__(X, y)  # get next cost
            self.costs.append(cost)

    def show_cost(self):
        # Plot costs
        plt.plot(self.cost_set)
        plt.title("Convergence of the Cost Function")
        plt.ylabel("J($\Theta$)")
        plt.xlabel("Iteration")
        plt.show()


# Initialize model, creating an instance of the class
model = NeuralNet()

# Train model

iris = datasets.load_iris()

regressors = iris.data
regressand = iris.target == 1

x_train, x_test, y_train, y_test = cross_validation.\
    train_test_split(regressors, regressand, test_size=0.2, random_state=1)

model.learn(x_train, y_train)

yhat = model.predict(x_test)[0]

roc_auc = metrics.roc_auc_score(y_test, yhat)

print(roc_auc)


def __sigmoid_activation__(X, theta):
    X = np.array(X)
    theta = np.array(theta)

    return 1 / (1 + np.exp(-np.dot(X, theta.T)))  # logistic sigmoid function

theta = np.ones(4)
X = range(4)

result = __sigmoid_activation__(X, theta)

print(result)