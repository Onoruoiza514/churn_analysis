import sys
from src.utils.loggers  import logger


def error_message_detail(error: Exception, error_detail=sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in file [{file_name}] "
        f"at line [{line_number}] "
        f"with message: {str(error)}"
    )


class CustomException(Exception):
    """
    Custom exception class that captures detailed error context
    including file name and line number.
    """

    def __init__(self, error: Exception, error_detail=sys):
        self.error_message = error_message_detail(error, error_detail)
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message
