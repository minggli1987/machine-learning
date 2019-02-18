import numpy as np


def taylor_exp(x, k=3):
    approx = 0

    for i in range(k):
        approx += np.power(x, i) / np.math.factorial(i)

    return approx

if __name__ == "__main__":
    val = 3
    for k in range(19):
        yhat = taylor_exp(val, k=k)
        y = np.exp(val)
        print(f"true exponential function: {y}; taylor approximation at {k}th order: {yhat}")
