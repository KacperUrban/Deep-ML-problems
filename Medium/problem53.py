import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
	Q = X @ W_q
	K = X @ W_k
	V = X @ W_v
	return Q, K, V

def softmax(score):
	if score.ndim == 1:
		e_x = np.exp(score - np.max(score))
		return e_x / e_x.sum()
	else:
		e_x = np.exp(score - np.max(score, axis=1, keepdims=True))
		return e_x / e_x.sum(axis=1, keepdims=True)

def self_attention(Q, K, V):
	d_k = K.shape[1]
	scores = Q @ K.T / np.sqrt(d_k)
	A = softmax(scores)
	attention_output = A @ V
	return attention_output
