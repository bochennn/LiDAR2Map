import os


def remove_clip_redundant_data(clip_path):
    clip_pose_path = os.path.join(clip_path, "pose_info")
    clip_tta_det_path = os.path.join(clip_path, "TTA_fused_for_tracking")
    clip_det_path = os.path.join(clip_path, "det_for_tracking")

    pose_file_names = sorted(os.listdir(clip_pose_path))
    tta_file_names = sorted(os.listdir(clip_tta_det_path))
    det_file_names = sorted(os.listdir(clip_det_path))

    assert len(tta_file_names) == len(det_file_names)
    if len(pose_file_names) > len(det_file_names):
        delta_num = len(pose_file_names) - len(det_file_names)
        print("pose file amount larger than det: {} > {}, about to remove {} frames from pose info".
              format(len(pose_file_names), len(det_file_names), delta_num))
        to_rm_file_names = list(set(pose_file_names) - set(det_file_names))
        for name in to_rm_file_names:
            target_file_path = os.path.join(clip_pose_path, name)
            os.system("rm {}".format(target_file_path))
    elif len(pose_file_names) < len(det_file_names):
        delta_num = len(det_file_names) - len(pose_file_names)
        print("det file amount larger than pose: {} > {}, about to remove {} frames from det and TTA_det".
              format(len(det_file_names), len(pose_file_names), delta_num))
        to_rm_file_names = list(set(det_file_names) - set(pose_file_names))
        for name in to_rm_file_names:
            target_det_file_path = os.path.join(clip_det_path, name)
            os.system("rm {}".format(target_det_file_path))
            target_TTA_det_file_path = os.path.join(clip_tta_det_path, name)
            os.system("rm {}".format(target_TTA_det_file_path))
    assert len(os.listdir(clip_pose_path)) == len(os.listdir(clip_det_path)) == len(os.listdir(clip_tta_det_path))


def remove_redundant_data(root_data_path):
    total_num = len(os.listdir(root_data_path))
    for idx, clip_name in enumerate(os.listdir(root_data_path)):
        if clip_name.startswith("clip_"):
            clip_path = os.path.join(root_data_path, clip_name)
            remove_clip_redundant_data(clip_path)
            print("progress: {}/{}".format(idx+1, total_num))


if __name__ == "__main__":
    root_data_path = "/data/sfs_turbo/wcp/robotaxi_clip_info/inference_ret_0514"
    remove_redundant_data(root_data_path)

