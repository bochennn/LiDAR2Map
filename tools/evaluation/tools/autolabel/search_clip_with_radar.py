import os
import argparse

from tqdm import tqdm


def search_clip_with_radar(root_path, out_name):
    clip_path_list = []
    clip_with_radar_path_list = []
    for root, dirs, files in tqdm(os.walk(root_path)):
        for directory in dirs:
            if directory.startswith("clip_"):
                clip_path = os.path.join(root, directory)
                clip_path_list.append(clip_path)
                if os.path.exists(os.path.join(clip_path, "radar0.json")):
                    clip_with_radar_path_list.append(clip_path)
    print("{} clips in total, {} clips with radar".format(len(clip_path_list), len(clip_with_radar_path_list)))
    with open("{}_all.txt".format(out_name), 'w') as f:
        for clip_path in clip_path_list:
            clip_name = os.path.basename(clip_path)
            f.write("{}\n".format(clip_name))
    with open("{}_radar.txt".format(out_name), 'w') as f:
        for clip_path in clip_with_radar_path_list:
            clip_name = os.path.basename(clip_path)
            f.write("{}\n".format(clip_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('out_name', type=str)
    args = parser.parse_args()
    search_clip_with_radar(args.data_path, args.out_name)
