# import json
import os.path
import time

import pandas as pd

from ....log_mgr import logger
from ...base_objs.base_clip_obj import ClipBase
from ..parsers.gt_parser import parse_json as normal_parser
from ..parsers.gt_parser import parse_obejct_list as dict_obejct_parser
from ..parsers.gt_parser import parse_visual_json as visual_parser
from ..parsers.gt_parser import parse_zdrive_anno as zdrive_parser
from .obstacle_frame_obj import ObstacleFrameObj


class ObstacleClipGt(ClipBase):
    def __init__(self, data_path):
        start_t = time.time()
        self.data_path = data_path
        self.max_allowed_gap = 3
        self.frame_objs = dict()
        super().__init__()
        self.add_track_tag()
        end_t = time.time()
        logger.info("time consumed by ObstacleClipGt.__init__: {:.2f}s, obj_num: {}".
                    format(end_t - start_t, self.data.shape[0]))

    def parse(self):
        try:
            if isinstance(self.data_path, dict):
                # bochen ADD: read from a dict object contains frames info
                # format: {ts1: {gt=json_obj, pd=json_obj}, ts2: ...}
                # where json_obj is the same content as read from example_data/eval_obstacle
                return dict_obejct_parser(self.data_path)
            elif self.data_path.endswith("json"):
                if "data_all" in os.path.basename(self.data_path):
                    return normal_parser(self.data_path)
                else:
                    return zdrive_parser(self.data_path)
            elif os.path.isdir(self.data_path):
                return visual_parser(self.data_path)
            else:
                raise NotImplementedError(
                    "parser for {} is not been implemented in {}".format(os.path.basename(self.data_path),
                                                                         __file__))
        except Exception as e:
            logger.error("Error during ObstacleClipGt.parse(), error: {}, data path: {}".format(e, self.data_path))
            import traceback
            logger.error(traceback.format_exc())

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
        max_allowed_gap = 3 if not hasattr(self, "max_allowed_gap") else getattr(self, "max_allowed_gap")
        data["category"] = pd.Categorical(data["category"])  # to speed up groupby method
        if "object_id" in data.columns:
            region_id = data.groupby(["object_id", "category"])["adapt_frame_seq"]. \
                transform(lambda x: ((x - x.shift(1)) > max_allowed_gap).cumsum())
            data["track_id"] = data["object_id"].astype(str) + "_" + data["category"].astype(str) + "_" + \
                region_id.astype(str)

    def get_frame_obj_by_ts(self, ts):
        if ts not in self.frame_objs:
            self.frame_objs[ts] = ObstacleFrameObj(self.data.loc[[ts]], ts)
        return self.frame_objs[ts]

    def to_visual_json(self, out_path):
        for ts in self.ts_list:
            self.get_frame_obj_by_ts(ts).to_visual_json(out_path)
