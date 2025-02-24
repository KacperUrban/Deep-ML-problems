def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	b = -(matrix[0][0] + matrix[1][1])
	c = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
	delta = b**2 - 4 * c

	eigenvalues = [(-b + delta**(1/2)) / 2, (-b - delta**(1/2)) / 2]
	return eigenvalues