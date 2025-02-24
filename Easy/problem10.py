def means(vec):
	sum_val = 0
	for elem in vec:
		sum_val += elem
	return sum_val / len(vec)

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    mean_vals = [means(feature) for feature in vectors]
    m = len(vectors[0])
    n = len(vectors)
    
    results = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(i, n):
            cov = 0
            for elem1, elem2 in zip(vectors[i], vectors[j]):
                cov += (elem1 - mean_vals[i]) * (elem2 - mean_vals[j])
            cov /= (m - 1)
            results[i][j] = cov
            results[j][i] = cov

    return results