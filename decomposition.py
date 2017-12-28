#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import cov, dot
from numpy.linalg import cholesky, eig, svd

m, n = 100, 20
M = np.random.rand(m, n)
# !!! strangely numpy.cov assumes by default observations in columns and
# variables in rows, rowvar=False turns it off.
Sigma = cov(M, rowvar=False)
# covariance matrix is always symmetric (i.e. A = A.T) and positive
# semi-definite (i.e. x.H * A * x >= 0)
# unitary matrix is A.T = inv(A) and Hermitan Matrix is symmetric complex
# matrix and A.H = A

# Cholesky decomposition factorizes a symmetric positive semi-definite matrix A
# into L so that A = L * L.H
L = cholesky(Sigma)
assert np.allclose(dot(L, L.T), Sigma)


# Eigen decomposition requires symmetric matrix A so that A.dot(v) = λ * v
# where λ is a scalar.
ev, eig = eig(Sigma)
# sort eigen values and vectors
idx = np.argsort(ev)[::-1]
ev = ev[idx]
eig = eig[:, idx]

for i, eigenvector in enumerate(eig.T):
    assert np.allclose(dot(Sigma, eigenvector), eigenvector * ev[i])

assert np.allclose(dot(Sigma, eig), eig * ev)

U, s, V = svd(Sigma, full_matrices=True)
# S is the singular value matrix, a diagonal matrix same shape as M
S = np.zeros_like(Sigma)
S[:n, :n] = np.diag(s)
assert np.allclose(Sigma, dot(U, dot(S, V)))

# eigen vectors from eig or svd have occasionally opposite sign
assert np.allclose(np.abs(U), np.abs(eig))
