import os
from functools import partial

from objects.base_objs.base_clip_obj import ClipBase
from objects.occupancy.objs.occupancy_frame_obj import OccupancyFrame
from objects.occupancy.parsers.attribute_tool import Attr
from objects.occupancy.parsers.voxel_parser import parser as voxel_parser
from objects.occupancy.parsers.lidar_seg_parser import parser as lidar_seg_parser
from log_mgr import logger


class OccupancyClipGt(ClipBase):
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.frame_objs = dict()
        super().__init__()

    def data_type_deduce(self, data_path):
        file_name = os.listdir(data_path)[0]
        if file_name.endswith("pkl"):
            logger.info("data_path: {}, deduced data type: {}".format(data_path, "voxel"))
            return voxel_parser
        elif file_name.endswith("bin"):
            logger.info("data_path: {}, deduced data type: {}".format(data_path, "lidar_seg"))
            return partial(lidar_seg_parser, lidar2voxel_config=self.config)

    def parse(self):
        parser = self.data_type_deduce(self.data_path)
        return parser(self.data_path)

    def get_frame_obj_by_ts(self, ts):
        if ts not in self.frame_objs:
            self.frame_objs[ts] = OccupancyFrame(self.data.loc[[ts]][Attr.labeled_voxel], ts)
