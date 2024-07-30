from collections import defaultdict


class ObstacleTrack:
    def __init__(self, track_id, track_frames, track_type):
        self.track_id = track_id
        self.data = track_frames
        self.track_type = track_type
        self.length = len(self.data)
        self.tp_dict = None

    @property
    def match_track_accessor(self):
        if self.track_type == "gt":
            return lambda match: match.get_pred_track_id()
        else:
            return lambda match: match.get_gt_track_id()

    def get_track_start_time(self):
        return self.data[0].gt_instant.get_ts()

    def get_track_end_time(self):
        return self.data[-1].gt_instant.get_ts()

    def get_track_frame_num(self):
        return len(self.data)

    def get_max_distance(self):
        return max([match.gt_instant.get_distance_to_ego() for match in self.data])

    def get_min_distance(self):
        return min([match.gt_instant.get_distance_to_ego() for match in self.data])

    def get_min_lat_distance(self):
        return round(min([match.gt_instant.get_lidar_pos_y() for match in self.data]), 2)

    def get_max_lat_distance(self):
        return round(max([match.gt_instant.get_lidar_pos_y() for match in self.data]), 2)

    def get_min_lon_distance(self):
        return round(min([match.gt_instant.get_lidar_pos_x() for match in self.data]), 2)

    def get_max_lon_distance(self):
        return round(max([match.gt_instant.get_lidar_pos_x() for match in self.data]), 2)

    def is_in_lat_roi(self):
        return -60 < self.data[0].gt_instant.get_lidar_pos_y() < 60 \
            or -60 < self.data[-1].gt_instant.get_lidar_pos_y() < 60

    def is_in_lon_roi(self):
        return -100 < self.data[0].gt_instant.get_lidar_pos_x() < 120 \
            or -100 < self.data[-1].gt_instant.get_lidar_pos_x() < 120

    def get_category(self):
        if self.track_type == "gt":
            return self.data[0].get_gt_category()
        else:
            return self.data[0].get_pred_category()

    def get_track_id(self):
        return self.track_id

    def get_frame_num(self):
        return self.length

    def init_tp_dict(self):
        tp_dict = defaultdict(int)
        for match in self.data:
            if match.is_tp():
                tp_dict[self.match_track_accessor(match)] += 1
        self.tp_dict = tp_dict

    def get_tp_dict(self):
        if self.tp_dict is None:
            self.init_tp_dict()
        return self.tp_dict

    def intersection(self, other_track):
        if self.tp_dict is None:
            self.init_tp_dict()
        return self.tp_dict.get(other_track.get_track_id(), 0)

    def get_track_meta_info(self):
        start_time = self.get_track_start_time()
        end_time = self.get_track_end_time()
        duration = round((end_time-start_time) / 1000, 2)
        meta_info = {"track_id": self.track_id,
                     "frame_num": self.get_frame_num(),
                     "duration": duration,
                     "lat_min": self.get_min_lat_distance(),
                     "lat_max": self.get_max_lat_distance(),
                     "lon_min": self.get_min_lon_distance(),
                     "lon_max": self.get_max_lon_distance()}
        return meta_info
