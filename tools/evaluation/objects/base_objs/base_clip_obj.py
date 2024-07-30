from abc import ABC, abstractmethod
import uuid

import pandas as pd

from ..obstacle.parsers.attribute_tool import ts_round
from ...log_mgr import logger


class ClipBase(ABC):
    def __init__(self):
        self.data, self.ts_list = self.parse()
        self.add_uuid()

    @abstractmethod
    def parse(self):
        pass

    def get_ts_list(self):
        return self.ts_list

    @abstractmethod
    def get_frame_obj_by_ts(self, ts):
        pass

    @staticmethod
    def ts_round(ts):
        return ts_round(ts)

    def add_uuid(self):
        if "uuid" not in self.data.columns:
            obj_unique_id = [str(uuid.uuid4()) for _ in range(len(self.data))]
            self.data["uuid"] = obj_unique_id

    def add_track_tag(self):
        if "object_id" not in self.data.columns or "category" not in self.data.columns:
            return
        """
            annotation rule: the maximum allowed track id miss is 3, to find out all tracks
            step 1: for each group of unique value (object_id, category), find gaps larger than 3 frames
            step 2: name each region by a unique id which separated by those gaps
            step 3: combine (object_id, category, region_id) as unique key for tracks
        """
        data = self.data
        max_allowed_gap = 0.3 if not hasattr(self, "max_allowed_gap") else getattr(self, "max_allowed_gap")
        data["category"] = pd.Categorical(data["category"])  # to speed up groupby method
        if "object_id" in data.columns:
            region_id = data.groupby(["object_id", "category"])["measure_ts"]. \
                transform(lambda x: ((x - x.shift(1)) > max_allowed_gap).cumsum())
            data["track_id"] = data["object_id"].astype(str) + "_" + data["category"].astype(str) + "_" + \
                region_id.astype(str)

    def get_frame_obj_by_nearest_ts(self, ts):
        nearest_ts = min(self.ts_list, key=lambda x: abs(x-ts))
        if abs(nearest_ts - ts) < 0.1:
            return self.get_frame_obj_by_ts(nearest_ts)
        else:
            logger.info("time gap between {} and {} is too large".format(nearest_ts, ts))
            return None
