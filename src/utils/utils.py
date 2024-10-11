from typing import Callable
from utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function."""

    def wrap(*args, **kwargs):
        try:
            return task_func(*args, **kwargs)
        except Exception as ex:
            log.exception("Exception occurred during task execution.")
            raise ex
        finally:
            log.info(f"Task completed. Check the logs folder for details.")

    return wrap
