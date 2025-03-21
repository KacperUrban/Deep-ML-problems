import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    # Your code here
    scores = np.array(scores)
    stable_scores = scores - np.max(scores)
    log_sum_exp = np.log(np.sum(np.exp(stable_scores)))
    return stable_scores - log_sum_exp