import numpy as np 
def pca(data: np.ndarray, k: int) -> np.ndarray:
	data = (data - data.mean(axis=0)) / data.std(axis=0)
	covariance_matrix = np.cov(data.T)
	eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
	eigs = sorted(zip(eigenvalues, eigenvectors.T), reverse=True)
	principal_components = np.array([vec for _,vec in eigs[:k]])
	return np.round(principal_components, 4)