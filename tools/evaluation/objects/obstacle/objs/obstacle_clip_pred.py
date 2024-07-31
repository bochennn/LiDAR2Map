import os
import time

import pandas as pd
import simplejson as json

# from ..parsers.fusion_ret_parser import parser as fusion_parser
# from ..parsers.radar_ret_parser import parser_for_raw as radar_driver_parser
# from ..parsers.radar_ret_parser import parser_for_processed as radar_per_parser
from ....log_mgr import logger
from ...base_objs.base_clip_obj import ClipBase
from ..parsers.detection_ret_parser import \
    parse_object_list as detection_dict_parser
from ..parsers.detection_ret_parser import parser as detection_parser
# from ..parsers.detection_2d_parser import parser as detection_2d_parser
from ..parsers.tracking_ret_parser import parser as tracking_parser
from .obstacle_frame_obj import ObstacleFrameObj


class ObstacleClipPred(ClipBase):
    def __init__(self, data_path, score_thr=0.3):
        start_t = time.time()
        self.data_path = data_path
        self.score_thr = score_thr
        self.frame_objs = dict()
        super().__init__()
        self.add_track_tag()
        end_t = time.time()
        logger.info("time consumed by ObstacleClipPred.__init__: {:.2f}s, obj_num: {}".
                    format(end_t - start_t, self.data.shape[0]))

    @staticmethod
    def parser_deduce(data_path):
        if isinstance(data_path, dict): # bochen ADD
            logger.info("deduced pred parser type: detection")
            return detection_dict_parser
        elif isinstance(data_path, list) or os.path.isdir(data_path):
            file_path_list = data_path if isinstance(data_path, list) else [os.path.join(data_path, file_name) for
                                                                            file_name in os.listdir(data_path)
                                                                            if file_name.endswith(".json")]
            for file_path in file_path_list:
                with open(file_path, 'r') as f:
                    anno_data = json.load(f)
                if len(anno_data) > 0:
                    obj = anno_data[0]
                    if all(["utm_position" in obj,
                            "utm_velocity" in obj,
                            "obj_id" in obj]):
                        logger.info("deduced pred parser type: tracking/fusion")
                        return tracking_parser
                    elif all(["psr" in obj,
                              "obj_score" in obj,
                              "obj_type" in obj,
                              "obj_sub_type" in obj]):
                        logger.info("deduced pred parser type: detection")
                        return detection_parser
                    elif all(["psr" in obj,
                              "category" in obj,
                              "obj_score" in obj]):
                        logger.info("deduced pred parser type: detection")
                        return detection_parser
                    else:
                        raise NotImplementedError(
                            "parser for {} is not been implemented in {}\n data path: {}".format(
                                os.path.basename(data_path),
                                __file__, data_path))
        else:
            raise NotImplementedError(
                "parser for {} is not been implemented in {}\n data path: {}".format(os.path.basename(data_path),
                                                                                     __file__, data_path))

    # @staticmethod
    # def parser_deduce(data_path):
    #     if os.path.isdir(data_path):
    #         file_path_list = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path)
    #                           if file_name.endswith(".json")]
    #         for file_path in file_path_list:
    #             with open(file_path, 'r') as f:
    #                 anno_data = json.load(f)
    #             if len(anno_data) > 0:
    #                 obj = anno_data[0]
    #                 if all(["utm_position" in obj,
    #                         "utm_velocity" in obj,
    #                         "obj_id" in obj]):
    #                     logger.info("data_path: {}\ndeduced pred parser type: tracking/fusion".format(data_path))
    #                     return tracking_parser
    #                 elif all(["psr" in obj,
    #                           "obj_score" in obj,
    #                           "obj_type" in obj,
    #                           "obj_sub_type" in obj]):
    #                     logger.info("data_path: {}\ndeduced pred parser type: detection".format(data_path))
    #                     return detection_parser
    #                 else:
    #                     raise NotImplementedError(
    #                         "parser for {} is not been implemented in {}\n data path: {}".format(
    #                             os.path.basename(data_path),
    #                             __file__, data_path))
    #     else:
    #         raise NotImplementedError(
    #             "parser for {} is not been implemented in {}\n data path: {}".format(os.path.basename(data_path),
    #                                                                                  __file__, data_path))

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
        max_allowed_gap = 10000 if not hasattr(self, "max_allowed_gap") else getattr(self, "max_allowed_gap")
        data["category"] = pd.Categorical(data["category"])  # to speed up groupby method
        if "object_id" in data.columns:
            region_id = data.groupby(["object_id", "category"])["adapt_frame_seq"]. \
                transform(lambda x: ((x - x.shift(1)) > max_allowed_gap).cumsum())
            data["track_id"] = data["object_id"].astype(str) + "_" + data["category"].astype(str) + "_" + \
                region_id.astype(str)

    def parse(self):
        parser = self.parser_deduce(self.data_path)
        return parser(self.data_path)

    def get_frame_obj_by_ts(self, ts):
        if ts not in self.frame_objs:
            self.frame_objs[ts] = ObstacleFrameObj(self.data.loc[[ts]], ts, self.score_thr)
        return self.frame_objs[ts]

    def to_visual_json(self, out_path):
        for ts in self.get_ts_list():
            frame_obj = self.get_frame_obj_by_ts(ts)
            frame_obj.to_visual_json(out_path)
