from functools import wraps
from time import time

from ..log_mgr import logger


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = time()
        result = f(*args, **kw)
        logger.info('func:{} consume: {}s'.format(f.__name__, round(time() - start, 2)))
        return result
    return wrap
