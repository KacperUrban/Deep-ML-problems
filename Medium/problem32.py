import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    n_samples, n_features = X.shape
    feature_list = [np.ones(n_samples)]

    for d in range(1, degree + 1):
        for comb in combinations_with_replacement(range(n_features), d):
            new_feature = np.prod(X[:, comb], axis=1)
            feature_list.append(new_feature)

    return np.column_stack(feature_list)