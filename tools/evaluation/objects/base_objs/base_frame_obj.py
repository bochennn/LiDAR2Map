from abc import ABC, abstractmethod


class FrameBase(ABC):
    def __init__(self, data, ts):
        self.data = data
        self.ts = ts

    def get_ts(self):
        return self.ts

    @abstractmethod
    def get_instant_objects(self):
        pass

    @abstractmethod
    def is_empty(self):
        pass
