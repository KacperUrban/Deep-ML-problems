import numpy as np
def min_max(data: np.ndarray) -> np.ndarray:
	xprim = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
	return np.round(xprim, 4)

def standarize(data: np.ndarray) -> np.ndarray:
	means = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	z = (data - means) / std
	return np.round(z, 4)

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray): # type: ignore
	# Your code here
	standardized_data = standarize(data)
	normalized_data = min_max(data)
	return standardized_data, normalized_data