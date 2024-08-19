import os
from datetime import datetime
import logging
import sys

__all__ = ["sys_logger"]

TASK_NAME = None
SYS_LOGGER = None


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


def init_logger(name="evaluation", logdir=None) -> bool:
    global SYS_LOGGER
    if SYS_LOGGER is not None:
        return False

    # init log file
    # root_path = os.path.dirname(os.path.dirname(__file__))
    # log_path = os.path.join(root_path, "log")
    now = datetime.now().strftime("%Y-%m%d-%H%M%S")
    global TASK_NAME
    TASK_NAME = now

    # init logger, stream handler and file handler
    logger = logging.getLogger(name)

    # set format and logging level for logger
    formatter = logging.Formatter(
        '[%(asctime)s][%(process)d][%(filename)-20s: %(lineno)-5d][%(levelname)s] %(message)s')
    # formatter = MultiLineFormatter(
    #     '[%(asctime)s][%(process)d][%(filename)-20s: %(lineno)-5d][%(levelname)s] %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        logger.addHandler(sh)

    if logdir is not None:
        os.makedirs(logdir, exist_ok=True)
        log_file = os.path.join(logdir, "{}_{}.log".format(name, now))

        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger.info("{} logger initialized, log file saved to {}".format(name, log_file))

    SYS_LOGGER = logger
    sys.modules.pop('tools.evaluation.log_mgr')
    return True

