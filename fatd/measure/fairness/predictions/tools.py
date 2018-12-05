import numpy as np

# Prediction accuracy
def prediction_accuracy(prediction_object):
    y_hat = prediction_object.y_hat
    y = prediction_object.y

    hits = np.sum(np.equal(y, y_hat))

    return hits/y.shape[0]
