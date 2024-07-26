
from typing import Dict, List

import numpy as np
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.nuscenes_dataset import \
    NuScenesDataset as _NuScenesDataset

from .evaluate import batch_iou_numpy
from .utils import rasterize_map


@DATASETS.register_module(force=True)
class NuScenesDataset(_NuScenesDataset):

    def prepare_train_data(self, index: int) -> Dict:
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_ann_info(self, index: int) -> Dict:
        info = self.data_infos[index]
        instance_masks = rasterize_map(info['lane_polygons'],
                                       (30, 60), (200, 400),
                                       lane_types=self.CLASSES, thickness=5)
        semantic_masks = np.vstack([
            ~np.any(instance_masks, axis=0, keepdims=True), instance_masks != 0])

        anns_results = dict(
            gt_semantic_seg=semantic_masks
        )
        return anns_results

    def get_data_info(self, index: int) -> Dict:

        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            cam2img = []
            lidar2cam_rts = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                # lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                cam2img.append(intrinsic)
                lidar2cam_rts.append(lidar2cam_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    cam2img=np.stack(cam2img),
                    lidar2cam=np.stack(lidar2cam_rts)
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def evaluate(self, results: List[Dict], logger=None, **kwargs):

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        pts_seg_iou, fusion_seg_iou = [], []
        for idx, single_res in enumerate(results):
            anno_info = self.get_ann_info(idx)
            gt_semantic_seg = anno_info['gt_semantic_seg']

            single_res.update(gt_semantic_seg=gt_semantic_seg,
                              pts_semantic_seg=single_res['pts_semantic_seg'].cpu().numpy())
            pts_seg_iou.append(batch_iou_numpy(single_res['pts_semantic_seg'], gt_semantic_seg))

            # if 'fusion_semantic_seg' in single_res:
            #     fusion_seg_iou.append(batch_iou_numpy(single_res['fusion_semantic_seg'], gt_semantic_seg))

        res = dict()
        for idx, cls_iou in enumerate(np.stack(pts_seg_iou).mean(axis=0)):
            res[f'pts_seg_iou_cls_{idx}'] = cls_iou

        # if len(fusion_seg_iou) > 0:
        #     for idx, cls_iou in enumerate(fusion_seg_iou):
        #         res[f'fusion_seg_iou_cls_{idx}'] = cls_iou

        return res
