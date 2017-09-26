import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel


def euclidean(Xa, Xb):
    """distance sparse matrix"""
    dist = cdist(Xa, Xb, metric='minkowski', p=2)
    return np.sqrt(np.sum(dist.diagonal()**2))


np.random.seed(0)

a = np.random.rand(3, 4)
b = np.random.normal(size=(3, 4))

assert np.isclose(np.linalg.norm(a - b), euclidean(a, b))


def radial_basis_function(Xa, Xb=None, lengthscale=1):
    """Gaussian Kernel."""
    if Xb is None:
        Xb = Xa
    dist = cdist(Xa, Xb, metric='minkowski', p=2)**2
    return np.exp(- dist / 2 / lengthscale**2)


x = radial_basis_function(a, b, lengthscale=1)
print(x)
y = rbf_kernel(a, b, gamma=.5)
print(y)


N = 100
epsilon = 1e-10
# TODO an uniform prior assumption but why?
X_test = np.linspace(-5, 5, N).reshape(-1, 1)
K_ss = radial_basis_function(X_test, lengthscale=.5)
# K_ss = rbf_kernel(X, gamma=.5)
# TODO !!! why do we need to take root of kernel matrix sigma?
# f ~ μ + L * N(0, σ)
L = np.linalg.cholesky(K_ss + epsilon * np.eye(N))
f_prior = 0 + np.dot(L, np.random.normal(loc=0, size=(N, 3)))
print(f_prior.shape, X_test.shape)
plt.plot(X_test, f_prior)
plt.axis([-5, 5, -3, 3])
plt.show()

# new evidence
X_train = np.arange(-4, 2, 1).reshape(-1, 1)
y_train = np.sin(X_train)

# kernal matrix of observed
K = radial_basis_function(X_train, lengthscale=.5)
L = np.linalg.cholesky(K + 5e-5 * np.eye(X_train.shape[0]))

# conditional probability of multivariate gaussian given f_prior
K_s = radial_basis_function(X_train, X_test, lengthscale=.5)
Lk = np.linalg.solve(L, K_s)
# so that Lk * L = K_s
print(Lk.shape)

# TODO !!! why lower triangular matrix L instead of whole kernel matrix?
# according to Ebden 2008:
# mu = K_s * inv(K) * y = Lk * L * inv(L * L.T) * y =
# Lk * (L * inv(L)) * inv(L.T) * y = Lk * I * (inv(L.T) * y)
# TODO need to check if matrix order of mupltication holds
mean = np.dot(Lk.T, np.linalg.solve(L, y_train)).reshape((N, ))

# sample from f ~ posterior given x_test points
# according to Ebden 2008:
# K_ss - K_s * inv(K) * K_s.T = K_ss - Lk * L * inv(L * L.T) * (Lk * L).T =
# K_ss - Lk * L * inv(L) * inv(L.T) * Lk.T * L.T =
# K_ss - Lk * I * I * Lk.T =
# K_ss - Lk * Lk.T
L = np.linalg.cholesky(K_ss + 1e-6 * np.eye(N) - np.dot(Lk.T, Lk))
f_posterior = mean.reshape(-1, 1) + \
              np.dot(L, np.random.normal(loc=0, size=(N, 3)))

S = np.diag(K_ss) - np.sum(Lk**2, axis=0)
std = np.sqrt(S)

# dots
plt.plot(X_train, y_train, 'bs', ms=8)
# posterier samples
plt.plot(X_test, f_posterior)
# 4 sigma confidence interval roughly 95%
plt.gca().fill_between(X_test.flat, mean-2*std, mean+2*std, color="#dddddd")
plt.plot(X_test, mean, 'b--', lw=2)
plt.axis([-5, 5, -3, 3])
plt.show()
