import numpy as np

def tanh(z):
	return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	input_sequence = np.array(input_sequence)
	Wx = np.array(Wx)
	Wh = np.array(Wh)
	for i in range(len(input_sequence)):
		initial_hidden_state = tanh(Wx @ input_sequence[i] + Wh @ initial_hidden_state + b)
	return initial_hidden_state