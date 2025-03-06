import numpy as np

def softmax(x):
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    else:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

def compute_qkv(X, W_q, W_k, W_v):
	Q = X.dot(W_q)
	K = X.dot(W_k)
	V = X.dot(W_v)
	return Q, K, V

def self_attention(Q, K, V):
	d_k = Q.shape[-1]
	score = Q@K.T / np.sqrt(d_k)
	attention_weights = softmax(score)
	output = attention_weights@V

	return output

def multi_head_attention(Q, K, V, n_heads):
	range_of_d_k = Q.shape[1] // n_heads

	heads = []
	for i in range(n_heads):
		if i == n_heads - 1:
			Q_i = Q[:, i * range_of_d_k:]
			V_i = V[:, i * range_of_d_k:]
			K_i = K[:, i * range_of_d_k:]
		else:
			Q_i = Q[:, i * range_of_d_k: (i + 1) * range_of_d_k]
			V_i = V[:, i * range_of_d_k: (i + 1) * range_of_d_k]
			K_i = K[:, i * range_of_d_k: (i + 1) * range_of_d_k]

		head_i = self_attention(Q_i, K_i, V_i)
		heads.append(head_i)

	return np.concatenate(heads, axis=1)