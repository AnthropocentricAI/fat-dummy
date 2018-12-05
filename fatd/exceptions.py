# Error = Exception
class Error(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, message):
        self.message = message
        # super().__init__(self.message)

    def __str__(self):
        return self.message

class MissingImplementationError(Error):
    """Exception raised for unimplemented functionality.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message=None):
        if message is None:
            self.message = (
                    'This is a default message.\n'
                    'This method/function has not been implemented yet.'
                    )
        else:
            self.message = message

class IncorrectShapeError(Error):
    """Exception raised for unimplemented functionality.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message=None):
        if message is None:
            self.message = (
                    'This is a default message.\n'
                    'This array has incorrect shape.'
                    )
        else:
            self.message = message
