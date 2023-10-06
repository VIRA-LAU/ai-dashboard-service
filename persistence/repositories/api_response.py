class ApiResponse:
    def __init__(self, success: int = -1, data: dict = None, error=None):
        """
                Defines the response shape
                :param success: A boolean that returns if the request has succeeded or not
                :param data: The model's response
                :param error: The error in case an exception was raised
                """
        self.success: int = success
        self.data: dict = data
        self.error: Exception = error.__str__() if error is not None else ''