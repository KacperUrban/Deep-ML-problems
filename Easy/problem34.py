import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	one_hot = np.eye(x.shape[0], dtype=int).tolist()
	one_hot_enc = [one_hot[elem] for elem in x]
	return one_hot_enc