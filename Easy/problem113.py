import numpy as np

def relu(x):
    return np.maximum(0.0, x)

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    # Your code here
    first_pass = relu(x @ w1)
    second_pass = relu(first_pass @ w2)
    return relu(x + second_pass)