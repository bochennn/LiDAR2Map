import numpy as np

from ..eval_tasks.sub_tasks.obstacle_eval_tracking import ObstacleEvalTrack
from ...log_mgr import logger


class HeadingAbnormalChecker:
    def __init__(self, match_pair_list):
        self.match_pair_list = match_pair_list

    @staticmethod
    def get_zero_score(values):
        value_array = np.array(values)
        return (value_array - np.mean(value_array)) / np.std(value_array)

    def search_abnormal_on_track(self, track_id, track):
        pred_instants = [match.pred_instant for match in track.data if match.pred_valid()]
        gt_yaw_degrees = [match.gt_instant for match in track.data if match.pred_valid]
        pred_yaw_degrees = [instant.get_utm_yaw_degree() for instant in pred_instants]
        zero_score = self.get_zero_score(pred_yaw_degrees)
        logger.info("track_id: {}\nyaw degrees: {}\nzero score: {}".format(track_id, pred_yaw_degrees, zero_score))

    def search_abnormal_on_self(self):
        try:
            tracks = ObstacleEvalTrack.extract_track_from_match(self.match_pair_list, "pred")
        except Exception as e:
            logger.error(e)
            logger.error("extract pred track from match pairs failed, try to extract gt tracks")
            tracks = ObstacleEvalTrack.extract_track_from_match(self.match_pair_list, "gt")
        for track_id, track in tracks.items():
            if track_id in ["9_Car_0"]:
                self.search_abnormal_on_track(track_id, track)

    def start(self):
        self.search_abnormal_on_self()
