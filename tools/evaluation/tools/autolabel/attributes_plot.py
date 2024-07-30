import os
from itertools import groupby
import json

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)
pd.set_option('display.width', None)
# pd.set_option('display.height', None)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from objects.obstacle.objs.obstacle_clip_pred import ObstacleClipPred


TRACK_LENGTH_THRS = 5


def get_utm_start_point(pose_info_path):
    sorted_file_names = sorted(os.listdir(pose_info_path))
    pose_file_path = os.path.join(pose_info_path, sorted_file_names[0])
    with open(pose_file_path, 'r') as f:
        pose_data = json.load(f)
    utm_start_point = [pose_data["lidar2world"][0][-1],
                       pose_data["lidar2world"][1][-1]]
    return utm_start_point


def extrack_tracks(obstacle_obj):
    tracks = []
    for track_id, track_data in obstacle_obj.data.groupby(["track_id"]):
        tracks.append((track_id, track_data))
    return tracks


def plot_on_track(track_id, track_data, start_point=None, out_path=None):
    if len(track_data) < TRACK_LENGTH_THRS or "Cone" in track_id:
        return
    print("track_id: {}, data: {}".format(track_id, track_data))
    if start_point is not None:
        utm_start_x, utm_start_y = start_point
        track_data["utm_x"] -= utm_start_x
        track_data["utm_y"] -= utm_start_y
    track_data["utm_yaw"] = np.degrees(track_data["utm_yaw"])
    track_data["yaw_rate"] = (track_data["utm_yaw"] - track_data["utm_yaw"].shift(-1)) / (track_data["measure_ts"] - track_data["measure_ts"].shift(-1))
    fig = plt.figure(figsize=(25, 10), constrained_layout=True)
    gs = GridSpec(6, 12, figure=fig)
    ax1 = fig.add_subplot(gs[0:3, :4])
    ax2 = fig.add_subplot(gs[3:, :4])
    ax3 = fig.add_subplot(gs[0:3, 4:8])
    ax4 = fig.add_subplot(gs[3:, 4:8])
    ax5 = fig.add_subplot(gs[0:3, 8:])
    ax6 = fig.add_subplot(gs[3:, 8:])

    sns.lineplot(track_data, x="frame_seq", y="utm_x", ax=ax1)
    ax1.set_title("utm_x")
    sns.lineplot(track_data, x="frame_seq", y="utm_y", ax=ax2)
    ax2.set_title("utm_y")
    sns.lineplot(track_data, x="frame_seq", y="utm_abs_vel_x", ax=ax3, color="orange")
    ax3.set_title("vel_x")
    sns.lineplot(track_data, x="frame_seq", y="utm_abs_vel_y", ax=ax4, color="orange")
    ax4.set_title("vel_y")
    sns.lineplot(track_data, x="frame_seq", y="utm_yaw", ax=ax5, color="red")
    ax5.set_title("yaw")
    sns.lineplot(track_data, x="frame_seq", y="yaw_rate", ax=ax6, color="red")
    ax6.set_title("yaw_rate")

    fig.suptitle(track_id)
    if out_path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(out_path, "{}.jpg".format(track_id)))
    plt.close(fig)

def plot_on_tracks(data_path, start_point=None, out_path=None):
    obstacle_obj = ObstacleClipPred(data_path)
    tracks = extrack_tracks(obstacle_obj)
    for track_id, track_data in tracks:
        plot_on_track(track_id, track_data, start_point, out_path)
    print("number of tracks: {}".format(len(tracks)))


if __name__ == "__main__":
    data_path = "/mnt/data/autolabel/velocity_gt_sample_clip/clip_1689126018500/refined_fusion"
    pose_info_path = "/mnt/data/autolabel/velocity_gt_sample_clip/clip_1689126018500/pose_info"
    out_path = "/mnt/data/autolabel/velocity_gt_sample_clip/clip_1689126018500/plots"
    plot_on_tracks(data_path, start_point=get_utm_start_point(pose_info_path))
