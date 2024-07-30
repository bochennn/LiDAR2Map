from abc import ABC, abstractmethod

from log_mgr import logger


class InstantBase:
    def __init__(self, data, ts):
        self.data = data
        self.ts = ts

    def get_ts(self, dtype=float):
        if dtype is float:
            return self.ts
        elif dtype is str:
            return "{:.6f}".format(self.ts)
        else:
            logger.warning("dtype {} not implemented, use {}".format(dtype, float))
            return self.ts

    def get_uuid(self):
        return self.getattr("uuid")

    def getattr(self, key, default=None):
        return getattr(self.data, key, default)

    def hasattr(self, key):
        return hasattr(self.data, key)

    def setattr(self, key, value):
        setattr(self.data, key, value)
