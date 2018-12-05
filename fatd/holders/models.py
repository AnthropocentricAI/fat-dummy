import abc
import inspect
import numpy as np

from fatd.exceptions import MissingImplementationError
from fatd.holders.exceptions import (
    PretrainedModelError,
    UntrainedModelError
)

def check_model_compatibility(object_class):
    message = []
    attributes = {'__init__':0, 'fit':2, 'predict':1}
    for a in attributes:
        if not hasattr(object_class, a):
            message.append('The class is missing \'{}\' attribute.'.format(a))
        else:
            params = len(inspect.signature(
                getattr(object_class, a)).parameters
                ) - 1
            if params < attributes[a]:
                message.append((
                    'The \'{}\' attribute of the class '
                    'has too few attributes ({}). '
                    'It needs to have at least {}.'
                        ).format(a, params, attributes[a]))
    if message:
        print('\n'.join(message))
        return False
    else:
        return True

class Models(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        raise MissingImplementationError()

    @abc.abstractmethod
    def fit(self, X, y):
        raise MissingImplementationError()

    @abc.abstractmethod
    def predict(self, X):
        raise MissingImplementationError()

def distance(x, new_point):
    return np.linalg.norm(x - new_point)
def point_distance(new_point, X):
    return np.apply_along_axis(distance, 1, X, new_point)
def multiple_points_distance(X, new_points):
    return np.apply_along_axis(point_distance, 1, new_points, X)

class KNN(Models):
    def __init__(self, k=3):
        self._training_data_X = None
        self._training_data_y = None
        self.unique_labels = None
        self._data_points_number = None
        self._k = k
        self._model_fitted = False

    def fit(self, X, y):
        if self._model_fitted:
            raise PretrainedModelError()
        else:
            self._training_data_X = X
            self._training_data_y = y
            self.unique_labels = np.unique(y)
            self._data_points_number = self._training_data_X.shape[0]
            self._model_fitted = True

    def clear(self):
        self._training_data_X = None
        self._training_data_y = None
        self.unique_labels = None
        self._data_points_number = None
        self._model_fitted = False

    def predict(self, X):
        if not self._model_fitted: raise UntrainedModelError()

        distances = multiple_points_distance(self._training_data_X, X)
        if self._k < self._data_points_number:
            knn = np.argpartition(distances, self._k)
            predictions = []
            for row in knn:
                close_labels = self._training_data_y[row[:self._k]]
                values, counts = np.unique(close_labels, return_counts=True)
                # predictions.append([values[np.argmax(counts)]])
                predictions.append(values[np.argmax(counts)])
            predictions = np.array(predictions)
        else:
            values, counts = np.unique(self._training_data_y, return_counts=True)
            # predictions = np.array([[values[np.argmax(counts)]]]*X.shape[0])
            predictions = np.array([values[np.argmax(counts)]]*X.shape[0])
        return predictions
