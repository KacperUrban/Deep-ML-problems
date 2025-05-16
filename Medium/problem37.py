import numpy as np

def calculate_correlation_matrix(X, Y=None):
    # Your code here
    if Y is not None:
        cov_matrix = np.cov(X.T, Y.T)[:X.shape[1], X.shape[1]:]
        std_X = np.std(X, axis=0, ddof=1)
        std_Y = np.std(Y, axis=0, ddof=1)
        corr_matrix = cov_matrix / np.outer(std_X, std_Y)
        return corr_matrix
    else:
        cov_matrix = np.cov(X, rowvar=False)
        std = np.std(X, axis=0, ddof=1)
        corr_matrix = cov_matrix / np.outer(std, std)
    return corr_matrix