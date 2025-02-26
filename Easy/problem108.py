import numpy as np

def entropy(list_of_elements: list) -> float:
	elements, counts = np.unique(list_of_elements, return_counts=True)
	probs = counts / np.sum(counts)
	entropy_value = 0.0
	for prob in probs:
		entropy_value -= prob*np.log2(prob)
	return entropy_value

def gini(list_of_elements: list) -> float:
	_, counts = np.unique(list_of_elements, return_counts=True)
	probs = counts / np.sum(counts)
	gini_value = 1.0
	for prob in probs:
		gini_value -= prob**2
	return gini_value

def disorder(apples: list) -> float:
	"""
	Compute the disorder in a basket of apples.
	"""
	return entropy(apples)