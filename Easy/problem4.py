def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	m, n = len(matrix), len(matrix[0])
	means = []
	if mode =='column':
		for i in range(m):
			sum_of_el = 0
			for j in range(n):
				sum_of_el += matrix[j][i]
			means.append(sum_of_el / m)
	elif mode == 'row':
		for i in range(m):
			sum_of_el = 0
			for j in range(n):
				sum_of_el += matrix[i][j]
			means.append(sum_of_el / n)
	else:
		raise ValueError("This mode is unsupported!")
	return means