import numpy as np

def mean(value):
    return np.mean(value)

def median(value):
    return np.median(value)

def threshold(value, lower, upper):
    return np.clip(value, a_min=lower, a_max=upper)
