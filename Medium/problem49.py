import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
	# Your code here
	x = x0
	m0 = np.zeros_like(x)
	v0 = np.zeros_like(x)

	for t in range(1, num_iterations + 1):
		gt = grad(x)
		m0 = beta1 * m0 + (1 - beta1) * gt
		v0 = beta2 * v0 + (1 - beta2) * gt**2

		m_hat = m0 / (1 - beta1**t)
		v_hat = v0 / (1 - beta2**t)

		x = x - learning_rate * m_hat / (np.sqrt(v_hat + epsilon))
	return x
