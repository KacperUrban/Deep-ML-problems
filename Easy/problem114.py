import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
	_, _, c = x.shape
	return [np.mean(x[:,:,i]) for i in range(c)]