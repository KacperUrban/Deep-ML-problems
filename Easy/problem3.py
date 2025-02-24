def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	if len(a) * len(a[0]) != new_shape[0] * new_shape[1]:
		return []

	flatten_matrix = [element for row in a for element in row]
	reshaped_matrix = [flatten_matrix[i*new_shape[1]:i*new_shape[1]+new_shape[1]] for i in range(new_shape[0])]
	return reshaped_matrix