import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Constructs a detailed error message with file name and line number.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in python script [{file_name}] "
        f"at line number [{exc_tb.tb_lineno}] "
        f"with error message: [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


# def divide(a, b):
#     try:
#         return a / b
#     except ZeroDivisionError as e:
#         raise CustomException(e, sys)


# if __name__ == "__main__":
#     # Ensure logs are shown in console for this standalone run
#     import logging as pylogging
#     pylogging.basicConfig(level=pylogging.INFO)

#     logging.info("Testing exception handling...")

#     try:
#         result = divide(10, 0)
#     except CustomException as ce:
#         logging.error(f"Handled custom exception: {ce}")
#         print("\n⛔ Custom Exception Occurred:")
#         print(ce)
