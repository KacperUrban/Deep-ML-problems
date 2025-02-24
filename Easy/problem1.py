def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
	if len(a[0]) != len(b):
		return -1

	result = []
	for i in range(len(a)):
		row_sum = 0
		for j in range(len(a[0])):
			row_sum += a[i][j] * b[j]
		result.append(row_sum)
	return result