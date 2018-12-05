import numpy as np

import fatd.transform.tools.training
import fatd.holders

# From data to model
class Data2Model(object):

    def __init__(self, splitting_function=None):
        self.training_indices = None
        self.test_indices = None

        if splitting_function is None:
            self.splitting_function = \
                lambda X, y: fatd.transform.tools.training.train_test_split(
                        X, y, train_share=.8, seed=42
                        )
        else:
            self.splitting_function = splitting_function

    # data_to_model
    def transform(self, data_object, model_object):
        self.training_indices, self.test_indices = \
                self.splitting_function(data_object.data, data_object.target)
        model_object.fit(data_object.data[self.training_indices],
                         data_object.target[self.training_indices])
        return model_object

# From model to predictions
class Model2Predictions(object):

    def __init__(self):
        pass

    # model_to_predictions
    def transform(self, model_object, data_object, data_to_model_object=None):
        #=None, ground_truth=None):
        #if isinstance(data_object, fatd.holders.Data):
        #    data_to_predict = self.transform_data_object()
        #else:
        #    data_to_predict = self.transform_data_array()

        labels_set = np.unique(model_object.unique_labels)
        if data_to_model_object is not None:
            test_indices = data_to_model_object.test_indices

            if len(test_indices) == 0 or test_indices is None:
                print('Missing test partition in the data_to_model object. Using the whole data set.')
                data_to_predict = data_object.data
                ground_truth_to_predict = data_object.target

            else:
                print('Using test partition of the data based on the data_to_model object.')
                data_to_predict = data_object.data[test_indices]
                ground_truth_to_predict = data_object.target[test_indices]
        else:
            print('Missing data_to_model object. Using the whole data set.')
            data_to_predict = data_object.data
            ground_truth_to_predict = data_object.target

        predictions = model_object.predict(data_to_predict)
        return fatd.holders.Predictions(predictions,
                                        data_to_predict,
                                        ground_truth_to_predict,
                                        labels_set)
