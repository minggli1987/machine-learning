#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from numpy import cov, dot
from scipy.linalg import cholesky, eig, svd

m, n = 100, 20
M = np.random.rand(m, n)
# !!! strangely numpy.cov assumes by default observations in columns and
# variables in rows, rowvar=False turns it off.
# UPDATE this is just definition of Cov.
Sigma = cov(M, rowvar=False)

# definition of covariance matrix is X @ X.T (after mean-centreing) where X
# consists of random variables (dims) in its ROWS instead of columns.
Sigma_rowvar = cov(M.T)
assert np.allclose(Sigma, Sigma_rowvar)
# covariance matrix is always symmetric (i.e. A = A.T) and positive
# semi-definite (i.e. x.H * A * x >= 0)
# unitary matrix is A.T = inv(A) and Hermitan Matrix is symmetric complex
# matrix and A.H = A


def compute_cov(a, rowvar=True):
    # ensure rows are random variables and columns observations.
    if not rowvar:
        a = a.T

    # mean center random variables at rows.
    mean = a.mean(axis=1)[:, None]
    # number of observations
    n = a.shape[1]
    # number of degree of freedom
    dof = n - 1
    x = a - mean
    return x @ x.T / dof


manual_Sigma = compute_cov(M, rowvar=False)
assert np.allclose(manual_Sigma, Sigma)

# Cholesky decomposition factorizes a symmetric positive semi-definite matrix A
# into L so that A = L * L.H, where L is lower triangular matrix
L = cholesky(Sigma, lower=True)
assert np.allclose(L @ L.T, Sigma)

# cholesky is a reasonable way to check singular matrix.
z = np.random.rand(20, 20)
phi = z @ z.T
phi[:, -2:] = 1.0
with np.testing.assert_raises(np.linalg.LinAlgError):
    np.linalg.cholesky(phi)

# eigen decomposition requires symmetric matrix A so that A.dot(v) = λ * v
# where λ is a scalar.
evalues, ev = eig(Sigma, right=True)
# sort eigen values and vectors
idx = np.argsort(evalues)[::-1]
ev = ev[:, idx]
evalues = evalues[idx]

for i, eigenvector in enumerate(ev.T):
    assert np.allclose(Sigma @ eigenvector, eigenvector * evalues[i])

# assert eigenvector definition
assert np.allclose(dot(Sigma, ev), evalues * ev)

U, s, V_h = svd(Sigma, full_matrices=True)
# S is the singular value matrix, a diagonal matrix same shape as M
S = np.zeros_like(Sigma)
S[:n, :n] = np.diag(s)
assert np.allclose(Sigma, U @ S @ V_h)

# eigen vectors from eig or svd have occasionally opposite sign
assert np.allclose(np.abs(U), np.abs(ev))
# singular values and eigen values can coincoide for square matrix
assert np.allclose(s, evalues)

# svd extends to non-symmetric matrix but still requires positive definiteness
# avoiding the dot product, singular values of M is square root of eigen
# values from M.T @ M
U, s, V_h = svd(M, full_matrices=True)
S = np.zeros_like(M)
S[:n, :n] = np.diag(s)
assert np.allclose(M, U @ S @ V_h)

evalues, ev = eig(M.T @ M)
# requires sorting eigenvectors
idx = np.argsort(evalues)[::-1]
evalues = evalues[idx]
ev = ev[:, idx]
assert np.allclose(s, np.power(evalues, .5))
