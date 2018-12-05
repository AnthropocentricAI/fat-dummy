import numpy as np

# Prediction confusion matrix
def prediction_confusion_matrix(predictions_object):
    y = predictions_object.y
    y_hat = predictions_object.y_hat
    unique_labels = predictions_object.unique_labels

    unique = np.unique(np.concatenate([y,y_hat, unique_labels]), axis=0)
    unique_n = unique.shape[0]
    matrix = np.zeros((unique_n, unique_n))

    for i, j in zip(y, y_hat):
        i_index = np.where(unique==i)[0][0]
        j_index = np.where(unique==j)[0][0]
        matrix[i_index, j_index] += 1

    return matrix
