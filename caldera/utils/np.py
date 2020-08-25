import numpy as np


def replace_nan_with_inf(m):
    np.putmask(m, np.isnan(m), np.full_like(m, np.inf))
    return m
