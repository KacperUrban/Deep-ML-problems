import numpy as np

def batch_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    # Your code here
    means_c = np.mean(X, axis=(0, 2, 3), keepdims=True)
    variance_c = np.var(X, axis=(0, 2, 3), keepdims=True) 

    X = (X - means_c) / np.sqrt(variance_c + epsilon)
    X = gamma * X + beta
    return X