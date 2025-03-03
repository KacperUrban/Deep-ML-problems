def det(matrix):
	return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
	inverse = [[matrix[1][1], -matrix[0][1]],[-matrix[1][0], matrix[0][0]]]
	det_val = det(matrix)

	for i in range(2):
		for j in range(2):
			inverse[i][j] = 1/det_val * inverse[i][j]
	return inverse