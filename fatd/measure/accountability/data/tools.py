import numpy as np

def class_count(data_holder):
    unique, counts = np.unique(data_holder.target, return_counts=True)
    return unique, counts
