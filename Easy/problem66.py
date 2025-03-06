def dot_product(vec1, vec2):
	result = 0
	for v1, v2 in zip(vec1, vec2):
		result += v1 * v2
	return result

def scalar_mul(scalar, vec2):
	for i in range(len(vec2)):
		vec2[i] = scalar * vec2[i]
	return vec2

def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	projection = dot_product(v, L) / dot_product(L, L)
	final_project = scalar_mul(projection, L)
	return final_project