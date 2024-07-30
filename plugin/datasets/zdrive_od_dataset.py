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

    def format_results(self, results: List[Dict]):
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        results_dict = dict()
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            pd_box3d = det['pts_bbox']['boxes_3d']
            pd_labels = det['pts_bbox']['labels_3d'].numpy()
            scores = det['pts_bbox']['scores_3d'].numpy()

            pd_box_center = pd_box3d.gravity_center.numpy()
            pd_box_dims = pd_box3d.dims.numpy()
            pd_box_yaw = pd_box3d.yaw.numpy()

            pd_box_list, gt_box_list = [], []
            for i in range(len(pd_box3d)):
                pd_box = dict(
                    obj_id=0, obj_type=0, obj_sub_type=0,
                    category=self.CLASSES[pd_labels[i]],
                    obj_score=scores[i],
                    psr=dict(
                        position=dict(
                            x=pd_box_center[i][0],
                            y=pd_box_center[i][1],
                            z=pd_box_center[i][2]),
                        scale=dict(
                            x=pd_box_dims[i][0],
                            y=pd_box_dims[i][1],
                            z=pd_box_dims[i][2]),
                        rotation=dict(x=0, y=0, z=pd_box_yaw[i]))
                )
                pd_box_list.append(pd_box)

            anno_info = self.get_ann_info(sample_id)
            gt_box3d = anno_info['gt_bboxes_3d']
            gt_names = anno_info['gt_names']

            gt_box_center = gt_box3d.gravity_center.numpy()
            gt_box_dims = gt_box3d.dims.numpy()
            gt_box_yaw = gt_box3d.yaw.numpy()

            for j in range(len(gt_box3d)):
                gt_box = dict(
                    measure_timestamp=0, track_id=0, obj_id=0,
                    obj_type=0, obj_sub_type=0, obj_score=1,
                    category=gt_names[j],
                    psr=dict(
                        position=dict(
                            x=gt_box_center[j][0],
                            y=gt_box_center[j][1],
                            z=gt_box_center[j][2]),
                        scale=dict(
                            x=gt_box_dims[j][0],
                            y=gt_box_dims[j][1],
                            z=gt_box_dims[j][2]),
                        rotation=dict(x=0, y=0, z=gt_box_yaw[j]))
                )
                gt_box_list.append(gt_box)
            results_dict[self.data_infos[sample_id]['timestamp']] = \
                dict(pd=pd_box_list, gt=gt_box_list)
        return results_dict

    def evaluate(self,
                 results: List[Dict],
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 out_dir=None,
                 **kwargs):

        results_dict = self.format_results(results, jsonfile_prefix)

        if jsonfile_prefix is not None:
            mmcv.mkdir_or_exist(jsonfile_prefix)
            res_path = f'{jsonfile_prefix}/results_zdrive_od.pkl'
            print('Results writes to', res_path)
            mmcv.dump(results_dict, res_path)

        return results_dict
