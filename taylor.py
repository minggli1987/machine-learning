import numpy as np


def taylor_exp(x, k=3):
	actual = np.exp(x)
	approx = 0

	for i in range(k):
		approx += np.power(x, i) / np.math.factorial(i)

	print(f"true exponential function: {actual}; taylor approximation at {k}th order: {approx}")


if __name__ == "__main__":
	for k in range(10):
		taylor_exp(3, k=k)
