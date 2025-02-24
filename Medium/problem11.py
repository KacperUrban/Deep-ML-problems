import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	x = np.zeros(A.shape[1])
	x_new = np.zeros_like(x)
	for _ in range(n):
		for i in range(A.shape[1]):
			sum_of_xes = 0
			for j in range(A.shape[0]):
				if i != j:
					sum_of_xes += A[i][j] * x[j]
			x_new[i] = round((b[i] - sum_of_xes) / A[i][i], 4)
		x[:] = x_new
	return x