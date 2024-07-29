# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp
from pathlib import Path

# from dataset_converters import indoor_converter as indoor
# from dataset_converters import kitti_converter as kitti
# from dataset_converters import lyft_converter as lyft_converter
from dataset_converters import nuscenes_converter
from dataset_converters import zdrive_converter
from dataset_converters.create_gt_database import (GTDatabaseCreater,
                                                   create_groundtruth_database)

# from mmengine import print_log

# from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos


def nuscenes_data_prep(root_path: Path,
                       out_dir: Path,
                       info_prefix: str,
                       version: str,
                       dataset_name: str = 'NuScenesDataset',
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    assert root_path.exists()
    out_dir.mkdir(parents=True, exist_ok=True)

    nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, info_prefix, version=version, max_sweeps=max_sweeps)

    # if version == 'v1.0-test':
    #     info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    #     update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_test_path)
    #     return

    # info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    # info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    # update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_train_path)
    # update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_val_path)
    # create_groundtruth_database(dataset_name, root_path, info_prefix,
    #                             f'{info_prefix}_infos_train.pkl')


def zdrive_data_prep(root_path: Path,
                     out_dir: Path,
                     info_prefix: str,
                     workers=1):

    batch_names = [
        'E03-CITY-20240702-undownloaded',
        'NON-E03-CITY-20240702-undownloaded',
    ]
    zdrive_converter.create_zdrive_infos(root_path, out_dir, info_prefix, batch_names, workers)

    return 

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='nuscenes', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=Path,
    default='./data/nuscenes',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=Path,
    default='./data/nuscenes',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='nuscenes')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--only-gt-database',
    action='store_true',
    help='''Whether to only generate ground truth database.
        Only used when dataset is NuScenes or Waymo!''')
args = parser.parse_args()

if __name__ == '__main__':
    # from mmengine.registry import init_default_scope
    # init_default_scope('mmdet3d')

    if args.dataset == 'nuscenes':
        if args.only_gt_database:
            create_groundtruth_database('NuScenesDataset', args.root_path,
                                        args.extra_tag,
                                        f'{args.extra_tag}_infos_train.pkl')
        else:
            train_version = f'{args.version}' # trainval
            nuscenes_data_prep(
                root_path=args.root_path,
                out_dir=args.out_dir,
                info_prefix=args.dataset,
                version=train_version,
                dataset_name='NuScenesDataset',
                max_sweeps=args.max_sweeps)
    elif args.dataset == 'zdrive':
        zdrive_data_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            info_prefix=args.dataset,
            workers=args.workers)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')