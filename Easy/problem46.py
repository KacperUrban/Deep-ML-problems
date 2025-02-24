import numpy as np
def precision(y_true, y_pred):
	tp = np.sum(((y_true == y_pred) & (y_true == 1)))
	fp = np.sum((y_true != y_pred) & (y_true == 0))
	if tp + fp == 0:
		return 0
	return tp / (tp + fp)