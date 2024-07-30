import tempfile
from pathlib import Path
from typing import Dict, List

import mmcv
import numpy as np
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset

from ..utils import transform_matrix


@DATASETS.register_module()
class ZDriveDataset(NuScenesDataset):

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidars']['lidar0']['filename'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] * 1e-6,
            lidar2ego=transform_matrix(
                info['lidars']['lidar0']['sensor2ego_translation'],
                info['lidars']['lidar0']['sensor2ego_rotation']
            )
        )
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        return input_dict

    def format_results(self, results, jsonfile_prefix=None):

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = Path(tempfile.TemporaryDirectory())
            jsonfile_prefix = tmp_dir.name / 'results'
        else:
            tmp_dir = None

        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):

            annos = []

            pd_info = dict(
                obj_id=0,
                obj_type=0,
                obj_sub_type=0,
                category=self.CLASSES[0],
                obj_score=0.8734740614891052,
                psr=dict(
                    position=dict(
                        x=6.148742198944092,
                        y=37.99754333496094,
                        z=-1.137405972480774
                    ),
                    scale=dict(
                        x=4.6380696296691895,
                        y=1.8572767972946167,
                        z=1.5237681865692139
                    ),
                    rotation=dict(
                        x=0,
                        y=0,
                        z=-1.4466677904129028
                    )
                )
            )

            gt_info = {
                "measure_timestamp": 1660121817.901000,
                "track_id": 1,
                "obj_id": 1,
                "obj_type": 0,
                "obj_sub_type": 7,
                "category": "Cyclist",
                "obj_score": 1,
                "psr": {
                    "position": {
                        "x": 11.4078,
                        "y": 25.7003,
                        "z": -1.987002
                    },
                    "scale": {
                        "x": 1.847,
                        "y": 0.955,
                        "z": 1.579
                    },
                    "rotation": {
                        "x": 0,
                        "y": 0,
                        "z": -1.485130433994752
                    }
                }
            }
        return None, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        results_dict = dict()

        return results_dict
