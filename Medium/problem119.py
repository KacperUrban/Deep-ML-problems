import numpy as np

def cramers_rule(A, b):
    A = np.array(A)
    b = np.array(b)
    n, m = A.shape
    x = []
    det_a = np.linalg.det(A)
    if not det_a:
        return -1
    for i in range(m):
        A_tmp = A.copy()
        A_tmp[:, i] = b
        x_i = np.linalg.det(A_tmp) / det_a
        x.append(x_i)
    return x
