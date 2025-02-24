import numpy as np

def divide_on_feature(X, feature_i, threshold):
	# Your code here
	boolean_vals = X[:,feature_i] >= threshold
	return X[boolean_vals], X[~boolean_vals]