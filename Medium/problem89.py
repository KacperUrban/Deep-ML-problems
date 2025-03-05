import numpy as np

def softmax(values):
	n = values.shape[0]
	softmax_val = np.zeros((n, n))
	for i, row in enumerate(values):
		sum_of_row = np.sum(np.exp(row))
		for j, elem in enumerate(row):
			softmax_val[i][j] = np.exp(elem) / sum_of_row
	return softmax_val

def pattern_weaver(n, crystal_values, dimension):
	# Your code here
	attention_scores = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			attention_scores[i][j] = (crystal_values[i] * crystal_values[j])/np.sqrt(dimension)
	x = softmax(attention_scores) @ crystal_values
	return np.round(x,3)