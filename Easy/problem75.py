def confusion_matrix(data):
	# Implement the function here
	confusion_matrix = [[0, 0],[0, 0]]
	for example in data:
		if example[0] == 1 and example[1] == 1:
			confusion_matrix[0][0] += 1
		elif example[0] == 0 and example[1] == 0:
			confusion_matrix[1][1] += 1
		elif example[0] == 0 and example[1] == 1:
			confusion_matrix[1][0] += 1
		else:
			confusion_matrix[0][1] += 1
	return confusion_matrix