#! /usr/bin/env python3
# -*- encoding: utf-8 -*-
from timeit import timeit
from functools import partial
import numpy as np
from scipy.linalg import solve_triangular
from numba import jit

X = np.random.rand(1000, 500)
y = np.random.rand(1000)

# solve X @ theta = y


# invert covariance matrix
def covariance_matrix_inversion(X, y):
    # X.T @ X @ theta = X.T @ y
    # theta = inv(X.T @ X) @ X.T @ y
    # inv actually calls LU factorization
    return np.linalg.inv(X.T @ X) @ X.T @ y


def cholesky_decomposition(X, y):
    # X.T @ X @ theta = X.T @ y
    # cholesky X.T @ X = cov = L @ L.T. where L is lower triangular
    # L @ L.T @ theta = X.T @ y
    # forward substitution L @ r = B where r = L.T @ theta and d = X.T @ y
    # L @ r = d solve for r
    # back substitution
    # L.T @ theta = r solve for theta
    L = np.linalg.cholesky(X.T @ X)
    d = X.T @ y
    # forward substitution
    r = solve_triangular(L, d, lower=True)
    # back substitution
    return solve_triangular(L.T, r, lower=False)


def qr_decomposition(X, y):
    # solve X @ theta = y
    # Q @ R @ theta = y where Q is orthogonal and Q is upper triangular
    # R @ theta = inv(Q) @ y
    # R @ theta = Q.T @ y
    # back substitution where d = Q.T @ y
    q, r = np.linalg.qr(X)
    d = q.T @ y
    return solve_triangular(r, d, lower=False)


def svd_decomposition(X, y):
    # solve X @ theta = y
    # X = U @ Σ @ V.T so U.T @ X @ V = Σ
    # minimize l2norm{y - X @ theta} = l2norm{U.T @ y - U.T @ X @ theta}
    #   because l2norm{U.T @ x} = l2norm{x} when U is orthogonal matrix
    # = l2norm{U.T @ y - (U.T @ X @ V) @ V.T @ theta} because V @ V.T = I
    # = l2norm{U.T @ y - Σ @ V.T @ theta}
    # when minimum U.T @ y = Σ @ V.T @ theta
    # theta = inv(V.T) @ inv(Σ) @ U.T @ y = V @ inv(Σ) @ U.T @ y
    # Moore-Penrose pseudo-inverse
    # theta = pinv(X) @ y
    return np.linalg.pinv(X) @ y


cov_inv_solution = partial(covariance_matrix_inversion, X, y)
cholesky_solution = partial(cholesky_decomposition, X, y)
qr_solution = partial(qr_decomposition, X, y)
svd_solution = partial(svd_decomposition, X, y)
print(timeit(cov_inv_solution, number=100))
print(timeit(cholesky_solution, number=100))
print(timeit(qr_solution, number=100))
print(timeit(svd_solution, number=100))

# jit compiling
cov_inv_solution = partial(jit(covariance_matrix_inversion), X, y)
cholesky_solution = partial(jit(cholesky_decomposition), X, y)
qr_solution = partial(jit(qr_decomposition), X, y)
svd_solution = partial(jit(svd_decomposition), X, y)
print(timeit(cov_inv_solution, number=100))
print(timeit(cholesky_solution, number=100))
print(timeit(qr_solution, number=100))
print(timeit(svd_solution, number=100))
