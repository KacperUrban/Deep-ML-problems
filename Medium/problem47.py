import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
  m = len(y)
	# Your code here
  if method == 'batch':
    for _ in range(n_iterations):
      predictions = np.dot(X, weights)
      errors = predictions - y
      gradient = np.dot(errors, X) / m
      weights -= learning_rate * 2 * gradient
  elif method == 'stochastic':
    for _ in range(n_iterations):
      for i in range(m):
        predictions = np.dot(X[i], weights)
        errors = predictions - y[i]
        gradient = errors * X[i]
        weights -= learning_rate * 2 * gradient
  elif method == 'mini_batch':
    for _ in range(n_iterations):
      for i in range(0, m, batch_size):
        xi = X[i:i+batch_size]
        yi = y[i:i+batch_size]
        predictions = np.dot(xi, weights)
        errors = predictions - yi
        gradient = np.dot(xi.T, errors) / batch_size
        weights -= learning_rate * 2 * gradient
  else:
    raise ValueError("Invalid method - must be batch, mini_batch, stochastic")
  return weights