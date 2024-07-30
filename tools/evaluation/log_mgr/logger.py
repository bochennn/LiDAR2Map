import os
from datetime import datetime
import logging


__all__ = ["sys_logger"]
INITIALIZED = False
TASK_NAME = None


class MultiLineFormatter(logging.Formatter):
    def get_header_length(self, record):
        """Get the header length of a given record."""
        return len(super().format(logging.LogRecord(
            name=record.name,
            level=record.levelno,
            pathname=record.pathname,
            lineno=record.lineno,
            msg='', args=(), exc_info=None
        )))

    def format(self, record):
        """Format a record with added indentation."""
        indent = ' ' * self.get_header_length(record)
        head, *trailing = super().format(record).splitlines(True)
        return head + ''.join(indent + line for line in trailing)


def init_logger(name="evaluation"):
    # init log file
    root_path = os.path.dirname(os.path.dirname(__file__))
    log_path = os.path.join(root_path, "log")
    os.makedirs(log_path, exist_ok=True)
    now = datetime.now().strftime("%Y-%m%d-%H%M%S")
    global TASK_NAME
    TASK_NAME = now
    log_file = os.path.join(log_path, "{}_{}.log".format(name, now))

    # init logger, stream handler and file handler
    logger = logging.getLogger(name)
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_file)

    # set format and logging level for logger
    formatter = logging.Formatter(
        '[%(asctime)s][%(process)d][%(filename)-20s: %(lineno)-5d][%(levelname)s] %(message)s')
    # formatter = MultiLineFormatter(
    #     '[%(asctime)s][%(process)d][%(filename)-20s: %(lineno)-5d][%(levelname)s] %(message)s')
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    fh.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        logger.addHandler(sh)
        logger.addHandler(fh)
    logger.info("{} logger initialized, log file saved to {}".format(name, log_file))
    return logger


if not INITIALIZED:
    sys_logger = init_logger()
    INITIALIZED = True
