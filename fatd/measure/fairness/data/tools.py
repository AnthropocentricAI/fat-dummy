import numpy as np

def feature_histogram(data_object, feature_index):
    if feature_index not in data_object.data_headers:
        raise ValueError('The index is not in the data object header.')

    counts, bins = np.histogram(data_object.data[feature_index])

    return bins, counts
