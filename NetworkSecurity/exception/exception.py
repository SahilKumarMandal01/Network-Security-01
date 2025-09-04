import sys
import logging

from NetworkSecurity.logging.logger import logging


class NetworkSecurityException(Exception):
    """Custom exception for network security errors with detailed traceback info."""

    def __init__(self, error_message: str, error_details: sys):
        super().__init__(error_message)     # preserve base Exception behavior
        self.error_message = error_message

        _, _, exc_tb = error_details.exc_info()
        self.lineno = exc_tb.tb_lineno if exc_tb else None
        self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else None
    
    def __str__(self):
        return (
            f"Error occured in script: [{self.file_name}] "
            f"at line: [{self.lineno}] "
            f"with message: [{self.error_message}]"
        )
    

if __name__ == "__main__":
    try:
        logging.info("Entering try block")
        a = 1 / 0 # This will trigger ZeroDivisionError
        print("This will not be printed", a)
    except Exception as e:
        raise NetworkSecurityException(e, sys)