import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from log_mgr import logger


class TrackMatchPair:
    def __init__(self, gt_track, pred_tracks=None):
        self.gt_track = gt_track
        self.pred_tracks = [] if pred_tracks is None else pred_tracks
        self.inter_map = dict()
        self.gt_num = self.gt_track.get_frame_num()
        self.fragment_thrs = 0.8
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.id_switch = 0
        self.id_tp = 0
        self.id_fp = 0
        self.id_fn = 0
        self.mt = 0  # mostly tracked trajectories, whose tp higher than 80%
        self.fm = 0  # pred fragment trajectories, whose tp lower than 80%
        self.true_pred_id = None

    def plot(self):
        gt_record = {"x": [match.gt_instant.get_lidar_pos_x() for match in self.gt_track.data],
                     "y": [match.gt_instant.get_lidar_pos_y() for match in self.gt_track.data],
                     "type": ["gt"] * len(self.gt_track.data)}
        pred_records = []
        for pred_track in self.pred_tracks:
            pred_records.append({"x": [match.pred_instant.get_lidar_pos_x() for match in pred_track.data],
                                 "y": [match.pred_instant.get_lidar_pos_y() for match in pred_track.data],
                                 "type": [pred_track.get_track_id()] * len(pred_track.data)})
        all_records = [pd.DataFrame(gt_record)]
        for pred_record in pred_records:
            all_records.append(pd.DataFrame(pred_record))
        plot_table = pd.concat(all_records)
        fig, ax = plt.subplots()
        sns.scatterplot(data=plot_table, x="x", y="y", hue="type", size="type", ax=ax)
        plt.show()

    def get_category(self):
        return self.gt_track.get_category()

    def update(self, pred_track, inter):
        self.pred_tracks.append(pred_track)
        self.inter_map[pred_track.get_track_id()] = inter

    def compute_fp_fm(self):
        for pred_track in self.pred_tracks:
            track_length = len(pred_track.data)
            track_fp = 0
            for match in pred_track.data:
                if match.is_fp():
                    self.fp += 1
                    track_fp += 1
            if (track_length - track_fp) / track_length < self.fragment_thrs:
                self.fm += 1

    def compute_tp_fn_ids_mt(self):
        previous_id = None
        track_length = len(self.gt_track.data)
        track_tp = 0
        for match in self.gt_track.data:
            if match.is_fn():
                self.fn += 1
                self.id_fn += 1
            if match.is_fp():
                self.id_fp += 1
            if match.is_tp():
                self.tp += 1
                track_tp += 1
                if previous_id is not None and match.get_pred_track_id() != previous_id:
                    self.id_switch += 1
                if match.get_pred_track_id() != self.true_pred_id:
                    self.id_fp += 1
                    self.id_fn += 1
                previous_id = match.get_pred_track_id()
        if track_tp / track_length > self.fragment_thrs:
            self.mt += 1

    def get_true_pred_track(self):
        if self.pred_tracks:
            self.true_pred_id, self.id_tp = max(self.gt_track.get_tp_dict().items(), key=lambda x: x[1])

    def compute(self, verbose=False):
        self.get_true_pred_track()
        self.compute_fp_fm()
        self.compute_tp_fn_ids_mt()
        if verbose:
            logger.info("pred: {}, gt: {}, tp: {}, fp: {}, fn: {}, "
                        "id_switch: {}, idtp: {}, idfp: {}, idfn: {}, "
                        "mt: {}, fm: {}".format(sum(len(pred_track.data) for pred_track in self.pred_tracks),
                                                len(self.gt_track.data), self.tp, self.fp, self.fn, self.id_switch,
                                                self.id_tp, self.id_fp, self.id_fn, self.mt, self.fm))
            self.plot()
        return self.tp, self.fp, self.fn, self.id_switch, self.id_tp, self.id_fp, self.id_fn, self.mt, self.fm
