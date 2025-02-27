import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	# Your code here, make sure to round
	m, n = X.shape
	theta = np.zeros((n, 1))
	y = np.expand_dims(y, axis=1)

	for _ in range(iterations):
		gradient = (X.T @ (X @ theta - y)) / m
		theta -= alpha * gradient
	return theta