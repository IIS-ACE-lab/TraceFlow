# services/side_channel_core/utils/error_handler.py
'''
This module captures exceptions in a function and logs the full traceback.
It is used to ensure that any errors that occur during the execution of a function are logged
for debugging purposes.


Author: Shraddha Mishra
'''

import functools
import traceback
from utils.logger import get_logger

def catch_exceptions(func):
    """
    Decorator to wrap a function in a try/except,
    log full traceback on failure, then re-raise.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error("Exception in %s: %s", func.__name__, e)
            logger.error(traceback.format_exc())
            raise
    return wrapper
