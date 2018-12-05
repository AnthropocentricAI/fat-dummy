import numpy as np

# Train confusion matrix
def training_confusion_matrix(model_object, data_to_model_object, data_object):
    training_indices = data_to_model_object.training_indices
    training_data = data_object.data[training_indices]
    training_labels = data_object.target[training_indices]

    training_predictions = model_object.predict(training_data)

    unique = np.unique(np.concatenate([training_predictions,training_labels]), axis=0)
    unique_n = unique.shape[0]
    matrix = np.zeros((unique_n, unique_n))

    for i, j in zip(training_labels, training_predictions):
        i_index = np.where(unique==i)[0][0]
        j_index = np.where(unique==j)[0][0]
        matrix[i_index, j_index] += 1

    return matrix

# Whole data confusion matrix
def data_confusion_matrix(model_object, data_object):
    data = data_object.data
    labels = data_object.target

    predictions = model_object.predict(data)

    unique = np.unique(np.concatenate([predictions,labels]), axis=0)
    unique_n = unique.shape[0]
    matrix = np.zeros((unique_n, unique_n))

    for i, j in zip(labels, predictions):
        i_index = np.where(unique==i)[0][0]
        j_index = np.where(unique==j)[0][0]
        matrix[i_index, j_index] += 1

    return matrix
