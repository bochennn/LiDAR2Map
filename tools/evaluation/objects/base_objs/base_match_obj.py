from abc import ABC, abstractmethod


class BaseMatch(ABC):
    def __init__(self, gt_instant, pred_instant, match_factor):
        self.gt_instant = gt_instant
        self.pred_instant = pred_instant
        self.match_factor = match_factor

    def gt_valid(self):
        return self.gt_instant is not None

    def pred_valid(self):
        return self.pred_instant is not None

    def match_valid(self):
        return self.gt_valid() and self.pred_valid()

    @abstractmethod
    def is_tp(self):
        pass

    @abstractmethod
    def is_fp(self):
        pass

    @abstractmethod
    def is_fn(self):
        pass
