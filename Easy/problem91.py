def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	# Your code here
	if len(y_true) != len(y_pred):
		raise ValueError("Lengths of y_true and y_pred must be the same")

	tp, fp, fn = 0, 0, 0

	for true, pred in zip(y_true, y_pred):
		if true == pred and true == 1:
			tp += 1
		elif true != pred and true == 0:
			fp += 1
		elif true != pred and true == 1:
			fn += 1

	precision = tp / (tp + fp)
	recall = tp / (tp + fn)

	f1 = 2 * (precision * recall) / (precision + recall)
	return round(f1,3)