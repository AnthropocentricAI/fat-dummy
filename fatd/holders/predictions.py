class Predictions(object):

    def __init__(self, y_hat, X=None, y=None, unique_labels=None):
        self.y_hat = y_hat
        self.X = X
        self.y = y
        self.unique_labels = unique_labels
