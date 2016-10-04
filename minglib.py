import statsmodels.api as sm
import numpy as np
from sklearn import linear_model

__author__ = 'Ming Li @ London, UK'


def forward_select(data, target, alpha=0.05, display=True):

    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response w

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """

    remaining = set([i for i in data.columns if i not in target.columns])
    selected_var = []
    current_score, best_new_score = 0.0, 0.0

    print('beginning forward stepwise variable selection...\n')
    while remaining and current_score == best_new_score:

        scores_with_candidates = []  # containing variables

        for candidate in remaining:

            X = sm.add_constant(data[selected_var + [candidate]]) # inherit variables from last step and try new ones
            reg = sm.OLS(target, X).fit()
            score, p = reg.rsquared, reg.pvalues[-1]  # r2 (changeable) and two-tailed p value of the candidate
            scores_with_candidates.append((score, p, candidate))

        scores_with_candidates.sort(reverse=False)  # order variables by score in ascending
        disqualified_candidates = [i for i in scores_with_candidates if ~(i[1] < alpha)]
        scores_with_candidates = [i for i in scores_with_candidates if i[1] < alpha]
        try:
            best_new_score, best_candidate_p, best_candidate = scores_with_candidates.pop()
        except IndexError:  # when no candidate passes significance test at critical value
            disqualified_score, disqualified_p, disqualified_candidate = disqualified_candidates.pop(0)
            remaining.remove(disqualified_candidate)  # remove the worst disqualified candidate
#            print(disqualified_score, disqualified_p, disqualified_candidate)
            continue  # continuing the while loop
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            current_score = best_new_score
            selected_var.append(best_candidate)
            if display:
                print(selected_var)

    model = sm.OLS(target, sm.add_constant(data[selected_var])).fit()

    print('forward stepwise selection completed...\n')

    return model, selected_var


def backward_select(data, target, alpha=0.05, display=True):
    """Linear model designed by backward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response w

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """

    selected_var = list(set([i for i in data.columns if i not in target.columns]))

    if display:
        print('beginning backward stepwise variable selection...\n')

    while selected_var:
        if display:
            print(selected_var)

        scores_with_candidates = []  # containing variables

        X = sm.add_constant(data[selected_var])  # inherit variables from last step and try new ones
        reg = sm.OLS(target, X).fit()
        score = reg.rsquared  # r2 (changeable) and two-tailed p value of the candidate
        p = reg.pvalues[1:]  # first p value belongs to constant
        for i, var in enumerate(selected_var):
            scores_with_candidates.append((p[i], score,  var))
        scores_with_candidates.sort(reverse=False)  # order variables by p value in ascending order
        disqualified_candidates = [i for i in scores_with_candidates if ~(i[0] < alpha)]
        try:
            scores_with_candidates.remove(disqualified_candidates.pop())
            selected_var = [i[2] for i in scores_with_candidates]

        except IndexError:  # when no candidate fails the significance test at critical value
            selected_var = [i[2] for i in scores_with_candidates]
            break

    model = sm.OLS(target, sm.add_constant(data[selected_var])).fit()

    print('\nbackward stepwise selection completed...\n')

    return model, selected_var


class GradientDescent(object):

    def __init__(self, alpha=.1, max_epochs=5000, conv_thres=.0001, display=False):

        self._alpha = alpha    # learning rate
        self._max_epochs = max_epochs  # max number of iterations
        self._conv_thres = conv_thres    # convergence threshold
        self._display = display
        self._multi_class = False
        self._sigmoid = None
        self._linear = None
        self.params = None
        self.X = None
        self.y = None

    def fit(self, X, y, model):

        self.X = np.array(X)
        self.y = np.array(y).reshape(len(y), 1)

        if isinstance(model, sm.OLS) or isinstance(model, linear_model.LinearRegression):
            self._linear = True
            if hasattr(model, 'coef_'):
                self.params = np.array(np.matrix(model.coef_))
            if hasattr(model, 'params'):
                self.params = np.array(np.matrix(model.params))

        if isinstance(model, linear_model.LogisticRegression):
            self._sigmoid = True
            if hasattr(model, 'coef_'):
                self.params = np.array(model.coef_)

            unique_classes = np.unique(y)
            n = len(unique_classes)
            if n < 2:
                raise ValueError("Optimiser needs samples of at least 2 classes"
                                 " in the data, but the data contains only one"
                                 " class: {0}".format(unique_classes[0]))
            if n == 2:
                self._multi_class = False
            else:
                self._multi_class = True

        return self

    def __partial_derivative_cost__(self, params, X, y):

        J = 0
        m = len(X)

        if self._linear:
            h = np.dot(X, params.T)     # GLM hypothesis in linear algebra representation

        if self._sigmoid:
            h = 1 / (1 + np.exp(-np.dot(X, params.T)))      # logistic (sigmoid) model hypothesis

        J = np.dot(X.T, (h - y)) / m        # partial_derivative terms for either linear or logistic regression
        return J.T  # J is a n-dimensioned vector

    def __cost_function__(self, params, X, y):

        J = 0
        m = len(X)

        if self._linear:
            h = np.dot(X, params.T)
            # GLM hypothesis in linear algebra representation
            J = (h - y) ** 2
            J /= (2 * m)
            return np.sum(J)

        if self._sigmoid:
            h = 1 / (1 + np.exp(-np.dot(X, params.T)))
            # logistic (sigmoid) model hypothesis
            J = -np.dot(np.log(h).T, y) - np.dot(np.log(1 - h).T, (1 - y))
            J /= m
            return np.sum(J)

    def __processing__(self, params, X, y):

        alpha = self._alpha

        count = 0  # initiating a count number so once reaching max iterations will terminate

        cost = self.__cost_function__(params, X, y)  # initial J(theta)
        prev_cost = cost + 10
        costs = [cost]
        thetas = [params]

        if self._display:
            print('beginning gradient decent algorithm...')

        while (np.abs(prev_cost - cost) > self._conv_thres) and (count <= self._max_epochs):
            prev_cost = cost
            params -= alpha * self.__partial_derivative_cost__(params, X, y)  # gradient descend
            thetas.append(params)  # restoring historic parameters
            cost = self.__cost_function__(params, X, y)  # cost at each iteration
            costs.append(cost)
            count += 1
            if self._display:
                print('iterations have been processed: {0}'.format(count))

        return params, costs

    def optimise(self):

        X = self.X
        y = self.y
        params = self.params

        if not self._multi_class:

            new_thetas, costs = self.__processing__(params, X, y)

            return new_thetas, costs

        if self._multi_class:

            n_samples, n_features = X.shape
            unique_classes = np.unique(y)
            master_params = np.empty(shape=(1, n_features))
            master_costs = list()

            for k, _class in enumerate(unique_classes):

                _y = np.array(y == _class).astype(int)  # one versus rest method handling multinominal classification
                _params = np.matrix(params[k])

                new_thetas, costs = self.__processing__(_params, X, _y)

                master_costs.append(costs)
                master_params = np.append(master_params, np.array(_params), axis=0)

            return master_params[1:], master_costs
