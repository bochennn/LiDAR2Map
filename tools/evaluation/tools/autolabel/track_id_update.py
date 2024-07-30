from objects.obstacle.objs.obstacle_clip_pred import ObstacleClipPred


def track_id_update(data_path, out_path):
    obstacle_obj = ObstacleClipPred(data_path)
    obstacle_obj.to_visual_json(out_path)


if __name__ == "__main__":
    data_path = "/mnt/data/autolabel/velocity_gt_sample_clip/velocity_yaw_plot_example/clip_1689297818500/tracking"
    out_path = "/mnt/data/autolabel/velocity_gt_sample_clip/velocity_yaw_plot_example/clip_1689297818500/track_updated"
    track_id_update(data_path, out_path)
