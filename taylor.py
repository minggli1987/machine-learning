"""
taylor expansion examples
https://en.wikipedia.org/wiki/Taylor_series

taylor approximation of exponential function

exp(x) ≅ x^0 / 0! + x^1 / 1! + x^2 / 2! + x^3 / 3! + ...

taylor approximation of logarithmic function

log(1 - x) ≅ -x - x^2/2 - x^3/3 - ...

"""

import numpy as np


def taylor_exp(x, k=3):
    approx = 0

    for i in range(k):
        approx += np.power(x, i) / np.math.factorial(i)

    return approx


def taylor_log_one_minus(x, k=3):
	approx = 0

	for i in range(1, k):
		approx -= np.power(x, i) / i

	return approx

if __name__ == "__main__":
    val = 3
    for k in range(19):
        yhat = taylor_exp(val, k=k)
        y = np.exp(val)
        print(f"true exponential function: {y}; taylor approximation at {k}th order: {yhat}")

    val = .3
    for k in range(19):
        yhat = taylor_log_one_minus(val, k=k)
        y = np.log(1 - val)
        print(f"true log function: {y}; taylor approximation at {k}th order: {yhat}")

