import copy
import numpy as np

from fatd.exceptions import (MissingImplementationError,
                             IncorrectShapeError)

def reshape_array(array, axis=0):
    if len(array.shape) == 1:
        if axis == 0:
            return array.reshape(1, array.shape[0])
        else:
            return array.reshape(array.shape[0], 1)
    else:
        return array

class Data(object):
    # TODO: This only supports numerical arrays
    def __init__(self, data_matrix, class_vector, data_headers=None, class_header=None):
        self.n_rows = None
        self.columns = None

        self.data = np.array(data_matrix)
        self.target = np.array(class_vector)
        self.update_size()

        if data_headers is None:
            self.data_headers = list(range(self.data.shape[1]))
        else:
            self.data_headers = data_headers

        if class_header is None:
            self.class_header = 0
        else:
            self.class_header = class_header

    def update_size(self):
        self.n_rows, self.columns = self.data.shape

    def append(self, array, target_array=None, axis=0, inplace=False):
        if axis == 0:
            if target_array is None:
                message = ('When adding new rows, you also need to add their '
                           'target value (target_array parameter).')
                raise ValueError(message)
            else:
                target_shape = target_array.shape
                if len(target_shape) == 1:
                    if target_shape[0] != array.shape[0]:
                        message = ('The target_array and data_array have sizes '
                                   'that do not match.')
                        raise IncorrectShapeError(message)
                else:
                    if target_shape[0] == 1:
                        if target_shape[1] != array.shape[0]:
                            message = ('The target_array and data_array have sizes '
                                       'that do not match.')
                            raise IncorrectShapeError(message)
                    else:
                        if target_shape[0] != array.shape[0]:
                            message = ('The target_array and data_array have sizes '
                                       'that do not match.')
                            raise IncorrectShapeError(message)

        else:
            if target_array is not None:
                message = ('When appending columns (axis=1) you cannot extend '
                           'the target vector as well.')
                raise ValueError(message)

        r_array = reshape_array(array, axis)
        a_array = np.append(self.data, r_array, axis)
        if inplace:
            self.data = a_array
            self.update_size()
            return None
        else:
            sc = copy.deepcopy(self)
            sc.data = a_array
            sc.update_size()
            return sc

    def delete(self, indices, axis=0, inplace=False):
        new_data = np.delete(self.data, indices, axis)
        if axis == 0:
            new_target = np.delete(self.target, indices, axis)
        else:
            new_target = self.target

        if inplace:
            self.data = new_data
            self.target = new_target
            self.update_size()
            return None
        else:
            sc = copy.deepcopy(self)
            sc.data = new_data
            sc.target = new_target
            sc.update_size()
            return sc

    def apply(self, transformation, indices=[], axis=0, inplace=False):
        if axis == 0:
            # Duplicate target if creating new rows
            if indices:
                new_target = np.concatenate([self.target, self.target[indices]])
            else:
                new_target = np.concatenate([self.target, self.target])
        else:
            new_target = self.target

        if indices:
            if axis == 0:
                # columns
                data_slice = self.data[indices,:]
            else:
                # rows
                data_slice = self.data[:,indices]
        else:
            data_slice = self.data

        new_slice = np.apply_along_axis(transformation, axis, data_slice)

        # TODO: Use `ewshape_array`
        d1 = new_slice.shape[0]
        d2 = 1 if len(new_slice.shape) == 1 else new_slice.shape[1]
        if axis == 1:
            # new_data = np.delete(self.data, indices, 1)
            new_data = np.append(self.data, new_slice.reshape(d1,d2), axis)
        else:
            # new_data = np.delete(self.data, indices, 0)
            new_data = np.append(self.data, new_slice.reshape(d2,d1), axis)

        if inplace:
            # replace the indices
            self.data = new_data
            self.target = new_target
            self.update_size()
            return None
        else:
            sc = copy.deepcopy(self)
            sc.data = new_data
            sc.target = new_target
            sc.update_size()
            return sc
