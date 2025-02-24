import math

def sigmoid(z: float) -> float:
	result = 1 / (1 + math.e**(-z))
	return round(result, 4)