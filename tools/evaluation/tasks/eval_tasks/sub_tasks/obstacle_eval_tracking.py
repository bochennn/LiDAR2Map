from itertools import groupby

import pandas as pd
from tabulate import tabulate

from ....objects.obstacle.objs.instant_track_match_pair import TrackMatchPair
from ....objects.obstacle.objs.Instant_track import ObstacleTrack
from ....log_mgr import logger


class ObstacleEvalTrack:
    def __init__(self, match_pair_list):
        self.match_pair_list = match_pair_list

    @staticmethod
    def extract_track_from_match(match_pair_list, target_type):
        def is_valid(match):
            if target_type == "gt":
                return match.gt_valid()
            else:
                return match.pred_valid()

        def get_track_id(match):
            if target_type == "gt":
                return match.get_gt_track_id()
            else:
                return match.get_pred_track_id()

        valid_matches = [match_pair for match_pair in match_pair_list if is_valid(match_pair)]
        valid_matches = sorted(valid_matches, key=get_track_id)
        tracks = dict()
        for track_id, track in groupby(valid_matches, key=get_track_id):
            tracks.update({track_id: ObstacleTrack(track_id, list(track), target_type)})
        return tracks

    def evaluating_track(self):
        match_pair_list = self.match_pair_list
        gt_tracks = self.extract_track_from_match(match_pair_list, "gt")
        try:
            pred_tracks = self.extract_track_from_match(match_pair_list, "pred")
        except TypeError:
            logger.warning("TypeError found when extract tracks from pred data, "
                           "evaluation of tracking will not be processed")
            return
        # initial track_match obj, 1 gt track can match with multiple pred tracks
        track_match_dict = {gt_track_id: TrackMatchPair(gt_track) for gt_track_id, gt_track in gt_tracks.items()}
        match_info = dict()  # pred_track_id: [matched_gt_track_id, match_ratio]
        """
        for each gt track:
        1. iterate each frame, count tp num and record the total tp num of each pred track
        2. for those pred track that partly matched with gt track, record as the matched track
        """
        for gt_track_id, gt_track in gt_tracks.items():
            tp_dict = gt_track.get_tp_dict()
            for pred_track_id, tp_num in tp_dict.items():
                match_ratio = tp_num / pred_tracks[pred_track_id].get_frame_num()
                if (match_info.get(pred_track_id) and match_ratio > match_info.get(pred_track_id)[1]) or \
                        (match_info.get(pred_track_id) is None):
                    match_info[pred_track_id] = [gt_track_id, match_ratio]
        for pred_track_id, (gt_track_id, match_ratio) in match_info.items():
            track_match_dict[gt_track_id].update(pred_tracks[pred_track_id], match_ratio)
        category_list = ['Car', 'Truck', 'Bus', 'Cyclist', 'Person', 'Cone', 'Unknown']
        columns = ["tp", "fp", "fn", "id_switch", "mota", "gt_mt", "gt_track_num", "pred_fm", "pred_track_num"]
        result_table = pd.DataFrame(index=category_list, columns=columns)
        result_table.fillna(0, inplace=True)
        for track_match in track_match_dict.values():
            tp, fp, fn, id_switch, idtp, idfp, idfn, mt, fm = track_match.compute()
            result_table.loc[track_match.get_category(), "tp"] += tp
            result_table.loc[track_match.get_category(), "fp"] += fp
            result_table.loc[track_match.get_category(), "fn"] += fn
            result_table.loc[track_match.get_category(), "id_switch"] += id_switch
            result_table.loc[track_match.get_category(), "gt_mt"] += mt
            result_table.loc[track_match.get_category(), "gt_track_num"] += 1
            result_table.loc[track_match.get_category(), "pred_fm"] += fm
            result_table.loc[track_match.get_category(), "pred_track_num"] += len(track_match.pred_tracks)
        result_table["mota"] = (1 - (result_table["fp"] + result_table["fn"] + result_table["id_switch"])
                                / (result_table["tp"] + result_table["fn"])) * 100
        print(tabulate(result_table, headers=result_table.columns))
        return result_table

    def run(self):
        return self.evaluating_track()
