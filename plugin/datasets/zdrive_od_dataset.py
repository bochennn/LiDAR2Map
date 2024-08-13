from logging import Logger
from typing import Dict, List

import mmcv
import numpy as np
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset

from ..utils import transform_matrix


@DATASETS.register_module()
class ZDriveDataset(NuScenesDataset):
    NameMapping = {
        'car': 'car',
        'pickup_truck': 'pickup_truck',
        'truck': 'truck',
        'construction_vehicle': 'construction_vehicle',
        'bus': 'bus',
        'tricycle': 'tricycle',
        'motorcycle': 'motorcycle',
        'bicycle': 'bicycle',
        'person': 'person',
        'traffic_cone': 'traffic_cone'
    }

    def __init__(self,
                 ann_files: List[str],
                 pipeline: List[Dict] = None,
                 data_root: str = None,
                 classes: List[str] = None,
                 load_interval: int = 1,
                 with_velocity: bool = False,
                 modality: Dict = None,
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = False,
                 test_mode: bool = False,
                 use_valid_flag: bool = True,
                 logger: Logger = None):
        self.logger = logger
        if isinstance(ann_files, list):
            ann_file, self.additional_ann = ann_files[0], ann_files[1:]
        else:
            ann_file, self.additional_ann = ann_files, []

        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_root=data_root,
            classes=classes,
            load_interval=load_interval,
            with_velocity=with_velocity,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            use_valid_flag=use_valid_flag)

    def log_info(self, *logs):
        if self.logger is not None:
            self.logger.info(' '.join([l if isinstance(l, str) else str(l) for l in logs]))

    def load_annotations(self, ann_file: str) -> List[Dict]:
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        self.log_info('Loading', self.ann_file)
        data_infos = data['infos']
        self.metadata = data['metadata']
        self.version = self.metadata['version']

        if len(self.additional_ann) > 0:
            for f in self.additional_ann:
                self.log_info('Loading', f)
                data_infos.extend(mmcv.load(open(f, 'rb'), file_format='pkl')['infos'])

        data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))
        self.log_info(f'Total infos', len(data_infos))
        data_infos = data_infos[::self.load_interval]
        self.log_info(f'Infos after interval', len(data_infos))
        return data_infos

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

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for i, cat in enumerate(gt_names_3d):
            _cat = self.NameMapping[cat] if cat in self.NameMapping else 'other'
            gt_names_3d[i] = _cat
            gt_labels_3d.append(self.CLASSES.index(_cat) if _cat in self.CLASSES else -1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

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

            # from ..utils import read_points_pcd, convert_points
            # import numpy as np
            # from visualization import show_o3d
            # points = read_points_pcd(self.data_infos[sample_id]['lidars']['lidar0']['filename'])
            # points = convert_points(points, transform_matrix(
            #         self.data_infos[sample_id]['lidars']['lidar0']['sensor2ego_translation'],
            #         self.data_infos[sample_id]['lidars']['lidar0']['sensor2ego_rotation']
            #     ))
            # pts_mask = (points[:, 0] > -1) & (points[:, 0] < 4) & (points[:, 1] > -1) & (points[:, 1] < 1)
            # show_o3d([points[~pts_mask]], [{'box3d': np.vstack([
            #     np.hstack([gt_box_center, gt_box_dims, gt_box_yaw[:, None]]),
            #     np.hstack([pd_box_center, pd_box_dims, pd_box_yaw[:, None]])]),
            #     'labels': np.hstack([np.zeros(len(gt_box_center)), np.ones(len(pd_box_center))]).astype(int),
            #     'texts': np.hstack([gt_names, np.asarray(self.CLASSES)[pd_labels]]),
            # }])

            results_dict[self.data_infos[sample_id]['timestamp']] = \
                dict(pred=pd_box_list, gt=gt_box_list)
        return results_dict

    def evaluate(self,
                 results: List[Dict],
                 logger: Logger = None,
                 jsonfile_prefix: str = None,
                 result_names: List = ['pts_bbox'],
                 out_dir: str = None,
                 **kwargs):

        results_dict = self.format_results(results)

        if jsonfile_prefix is not None:
            mmcv.mkdir_or_exist(jsonfile_prefix)
            res_path = f'{jsonfile_prefix}/results_zdrive_od.pkl'
            logger.info('Results writes to', res_path)
            mmcv.dump(results_dict, res_path)

        from tools.evaluation.config import ConfigParser
        from tools.evaluation.tasks import ObstacleEval
        config = ConfigParser.parse('3d_object')
        config.update(process_num=8,
                      gt_data_path=results_dict,
                      pred_data_path=results_dict)
        od_eval = ObstacleEval(config)
        od_eval.start()

        eval_metrics = dict()
        for cat_id in range(len(od_eval.category)):
            eval_metrics[f'AP@50/{od_eval.category[cat_id]}'] = od_eval.bbox_eval_result[
                cat_id + cat_id * len(od_eval.distance)]['AP@50']

        return eval_metrics
