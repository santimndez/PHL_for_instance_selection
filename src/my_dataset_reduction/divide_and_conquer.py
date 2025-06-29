import numpy as np
from sklearn.model_selection import train_test_split

def divide_and_conquer_reduction(X, y, threshold, reduce_func, seed=None, **args):
    """
    Perform divide and conquer reduction on the dataset.

    Parameters:
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Target vector.
    threshold (int): Threshold to apply the reduction function or split the dataset in two.
    reduce_func (callable): data reduction function.
    seed (int, optional): Random seed for reproducibility.
    **args: Additional arguments for reduce_func.

    Returns:
    np.ndarray: Reduced feature matrix.
    np.ndarray: Reduced target vector.
    """
    if len(X) <= threshold:
        return reduce_func(X, y, **args)

    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)
    left_X, left_y = divide_and_conquer_reduction(Xa, ya, threshold, reduce_func, seed+1 if seed is not None else None, **args) # Vary the seed for each split
    right_X, right_y = divide_and_conquer_reduction(Xb, yb, threshold, reduce_func, seed+2 if seed is not None else None, **args)

    return np.vstack((left_X, right_X)), np.concatenate((left_y, right_y))

def divide_and_conquer_scores(X, y, threshold, score_func, seed=None, **args):
    """
    Perform divide and conquer reduction on the dataset and return scores in the original order.

    Parameters:
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Target vector.
    threshold (int): Threshold for the number of samples in a subset.
    score_func (callable): Function to compute scores on a subset.
    seed (int, optional): Random seed for reproducibility.
    **args: Additional arguments for the score_func.

    Returns:
    np.ndarray: Scores of each sample, in the same order as X.
    """
    n = len(X)
    if n <= threshold:
        return score_func(X, y, **args)

    # Use indices to preserve original order
    indices = np.arange(n)
    idx_a, idx_b = train_test_split(indices, test_size=0.5, stratify=y, random_state=seed)

    Xa, ya = X[idx_a], y[idx_a]
    Xb, yb = X[idx_b], y[idx_b]

    scores = np.zeros(n, dtype=float)
    scores[idx_a] = divide_and_conquer_scores(Xa, ya, threshold, score_func, seed+1 if seed is not None else None, **args) # Vary the seed for each split
    scores[idx_b] = divide_and_conquer_scores(Xb, yb, threshold, score_func, seed+2 if seed is not None else None, **args)

    return scores