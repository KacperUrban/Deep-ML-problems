import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    """
    Solve the system Ax = b using the Conjugate Gradient method.

    :param A: Symmetric positive-definite matrix
    :param b: Right-hand side vector
    :param n: Maximum number of iterations
    :param x0: Initial guess for solution (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x
    """
    if x0 is None:
        x = np.zeros_like(b, dtype=np.float64)
    else:
        x = x0

    r = b - A @ x
    p = r.copy()
    
    for _ in range(n):
        Ap = A @ p
        alpha = (r.T @ r) / (p.T @ Ap)
        x = x + alpha * p
        r_prev = r
        r = r - alpha * Ap

        if np.linalg.norm(r) < tol:
            return x
        
        beta = (r.T @ r) / (r_prev.T @ r_prev)
        p = r + beta * p

    return x