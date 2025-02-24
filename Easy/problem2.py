def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
	matrix_result = [[0 for _ in range(len(a))] for _ in range(len(a[0]))]
	for i in range(len(a)):
		for j in range(len(a[0])):
			matrix_result[j][i] = a[i][j]
	return matrix_result