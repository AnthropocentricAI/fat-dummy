import numpy as np

# Train accuracy
def training_accuracy(model_object, data_to_model_object, data_object):
    training_indices = data_to_model_object.training_indices
    training_data = data_object.data[training_indices]
    training_labels = data_object.target[training_indices]

    training_predictions = model_object.predict(training_data)

    hits = np.sum(np.equal(training_labels, training_predictions))

    return hits/training_predictions.shape[0]

# Whole data accuracy
def data_accuracy(model_object, data_object):
    data = data_object.data
    labels = data_object.target

    predictions = model_object.predict(data)

    hits = np.sum(np.equal(labels, predictions))

    return hits/predictions.shape[0]
