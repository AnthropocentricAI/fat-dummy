import fatd.exceptions

class PretrainedModelError(fatd.exceptions.Error):
    """Exception raised for trying to retrain a pre-trained model.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message=None):
        if message is None:
            self.message = (
                    'This is a default message.\n'
                    'This model has already been trained. '
                    'Try clearing it with \'.clear\' before training.'
                    )
        else:
            self.message = message


class UntrainedModelError(fatd.exceptions.Error):
    """Exception raised for trying to retrain a pre-trained model.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message=None):
        if message is None:
            self.message = (
                    'This is a default message.\n'
                    'This model has not been trained. '
                    'Try training it with \'.train\' '
                    'before calling \'.predict\'.'
                    )
        else:
            self.message = message
