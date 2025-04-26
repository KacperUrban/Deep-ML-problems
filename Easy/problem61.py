import numpy as np

def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if (fp + tp) == 0:
        return 0
    else:
        precision = tp / (fp + tp)

    if (fn + tp) == 0:
        return 0
    else:
        recall = tp / (fn + tp)

    f_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return np.round(f_score, 3)