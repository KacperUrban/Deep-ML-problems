import numpy as np
import math

def make_prediction_adaboost(X_feature, polarity, threshold):
	if polarity == 1:
		return np.array([1 if x >= threshold else -1 for x in X_feature])
	else:
		return np.array([1 if x < threshold else -1 for x in X_feature])

def compute_weighted_error(X, y, weights, polarity, threshold):
	preds = make_prediction_adaboost(X, polarity, threshold)
	error = np.sum(weights * (preds != y))
	return error

def adaboost_fit(X, y, n_clf):
	n_samples, n_features = np.shape(X)
	w = np.full(n_samples, (1 / n_samples))
	clfs = []

	# Your code here
	for _ in range(n_clf):
		best_polarity, best_threshold, feature_index, best_alpha, min_error = None, None, None, None, float('inf')
		for n_feat in range(n_features):
			X_col = X[:, n_feat]
			column_vals_unique = np.unique(X_col)
			for threshold in column_vals_unique:
				polarity = 1
				error = compute_weighted_error(X_col, y, w, polarity, threshold)
				if error > 0.5:
					error = 1 - error
					polarity = -1

				alpha = 0.5 * math.log((1 - error) / (error + 1e-10))
				
				if error < min_error:
					best_polarity = polarity
					best_threshold = threshold
					feature_index = n_feat
					best_alpha = alpha
					min_error = error

		clfs.append({
			'polarity' : best_polarity,
			'threshold' : best_threshold.item(),
			'feature_index' : feature_index,
			'alpha' : best_alpha,
			})
		predictions = make_prediction_adaboost(X[:, feature_index], best_polarity,best_threshold)
		w = w * np.exp(-best_alpha * y * predictions)
		w /= np.sum(w)

	return clfs