import numpy as np
from scipy.stats import mode


def mode_range(a, axis=0, tol=1E-3):
    """Find the mode of values to within a certain range"""
    a_trunc = a // tol
    vals, counts = mode(a_trunc, axis)
    mask = (a_trunc == vals)
    # mean of each row
    return np.sum(a * mask, axis) / np.sum(mask, axis)
