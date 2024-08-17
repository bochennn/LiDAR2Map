from pathlib import Path
from typing import Any, Dict, Tuple


import numpy as np
from mmcv.image import imnormalize
from mmdet3d.core.points import get_points_type
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines.loading import \
    LoadAnnotations3D as _LoadAnnotations3D
from mmdet3d.datasets.pipelines.loading import \
    LoadPointsFromFile as _LoadPointsFromFile
from mmdet3d.datasets.pipelines.loading import \
    LoadPointsFromMultiSweeps as _LoadPointsFromMultiSweeps
from PIL import Image

from ...utils import convert_points, rasterize_map, load_points


@PIPELINES.register_module(force=True)
class LoadPointsFromFile(_LoadPointsFromFile):

    def __init__(
        self,
        convert_ego: bool = False,
        coord_type: str = 'LIDAR',
        **kwargs
    ):
        super(LoadPointsFromFile, self).__init__(coord_type=coord_type, **kwargs)
        self.convert_ego = convert_ego

    def __call__(self, results):

        pts_filename = results['pts_filename']
        if not Path(pts_filename).exists():
            print(f'pts filepath {pts_filename} not exists!!!')
            return None

        points = load_points(pts_filename, self.load_dim)
        points = points[:, self.use_dim]

        attribute_dims = None
        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if 'lidar2ego' in results and self.convert_ego:
            points = convert_points(points, results['lidar2ego'])

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        return results


@PIPELINES.register_module(force=True)
class LoadPointsFromMultiSweeps(_LoadPointsFromMultiSweeps):

    def __init__(
        self,
        sweep_indices: Tuple[int] = [-2, -1, 0],
        use_dim: Tuple[int] = [0, 1, 2, 3, 4],
        convert_ego: bool = False,
        coord_type: str = 'LIDAR',
        **kwargs: Dict
    ):
        self.sweep_indices = sweep_indices
        self.convert_ego = convert_ego
        self.coord_type = coord_type
        super(LoadPointsFromMultiSweeps, self).__init__(
            sweeps_num=len(sweep_indices),
            use_dim=use_dim, **kwargs)

    def __call__(self, results: Dict):
        sweep_points_list, ts = [], results['timestamp']

        for i in self.sweep_indices:
            adj_sweep = results['sweeps'].get(i, 0)
            points_sweep = load_points(adj_sweep['filename'])[:, self.use_dim]
            points_sweep = convert_points(points_sweep, adj_sweep['ego2ref'] @ results['lidar2ego'])
            points_sweep[:, self.time_dim] = ts - adj_sweep['timestamp'] * 1e-6
            sweep_points_list.append(points_sweep)
        points = np.vstack(sweep_points_list)

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=None)
        results['points'] = points
        return results


@PIPELINES.register_module()
class PrepareImageInputs(object):

    def __init__(
        self,
        pre_scale: Tuple[float, float],
        pre_crop: Tuple[int, int],
        bgr_mean: Tuple[float, float] = (103.53, 116.28, 123.675),
        bgr_std: Tuple[float, float] = (57.375, 57.12,58.395),
        rand_scale: Tuple[float, float] = (1.0, 1.0),
        rand_rotation: Tuple[float, float] = (0.0, 0.0),
        rand_flip: bool = False
    ):
        self.pre_scale = pre_scale
        self.pre_crop = pre_crop
        self.bgr_mean = np.array(bgr_mean)
        self.bgr_std = np.array(bgr_std)
        self.rand_scale = rand_scale
        self.rand_rotation = rand_rotation
        self.rand_flip = rand_flip

    def preprocess(self, img: np.ndarray, img_shape: Tuple[int, ...]):
        # TODO: check if correct
        pil_img = Image.fromarray(img.astype('uint8'), mode='RGB')
        pil_img = pil_img.resize((int(img_shape[1] * self.pre_scale[1]),
                                  int(img_shape[0] * self.pre_scale[0])))
        pil_img = pil_img.crop((
            0, 0, pil_img.width + self.pre_crop[1], pil_img.height + self.pre_crop[0]))
        return pil_img

    def sample_augmentation(self, img_shape: Tuple[int, int]):
        W, H = img_shape
        rescale_ratio = np.random.uniform(*self.rand_scale)
        scale = (int(W * rescale_ratio), int(H * rescale_ratio))
        crop_h = int(np.random.uniform(*self.rand_scale) * scale[1]) - H
        crop_w = int(np.random.uniform(0, max(0, scale[0] - W)))
        crop = (crop_w, crop_h, crop_w + W, crop_h + H)
        rotate = np.random.uniform(*self.rand_rotation)
        flip = False
        if self.rand_flip and np.random.choice([0, 1]):
            flip = True
        return rescale_ratio, scale, crop, rotate, flip

    def img_transform(
        self,
        img: Image.Image,
        rotation,
        translation,
        resize,
        resize_dims,
        crop,
        rotate,
        flip: bool = False
    ):
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= crop[:2]
        if flip:
            A = np.asarray([[-1, 0], [0, 1]])
            b = np.asarray([crop[2] - crop[0], 0])
            rotation = A @ rotation
            translation = A @ translation + b
        theta = rotate / 180 * np.pi
        A = np.asarray([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = np.asarray([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A @ -b + b
        rotation = A @ rotation
        translation = A @ translation + b

        return img, rotation, translation

    def color_normalize(self, img: Image.Image) -> np.ndarray:
        # Convert to float after channel conversion to ensure efficiency
        img = np.asarray(img)
        # Normalization.
        assert len(img.shape) == 3 and img.shape[2] == 3, (
            'If the mean has 3 values, the input tensor '
            'should in shape of (H, W, 3), but got the '
            f'tensor with shape {img.shape}')
        return imnormalize(img.astype(np.float32),
                           self.bgr_mean, self.bgr_std, to_rgb=False)

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:

        new_imgs = []
        transforms = []
        for ori_img in results['img']:
            pil_img = self.preprocess(ori_img, results['img_shape'])
            
            rescale_ratio, rescale, crop, rotate, flip = \
                self.sample_augmentation(pil_img.size)
            post_rot = np.eye(2) * np.asarray(self.pre_scale)
            post_tran = np.asarray(self.pre_crop)

            new_img, rotation, translation = self.img_transform(
                pil_img,
                post_rot,
                post_tran,
                resize=rescale_ratio,
                resize_dims=rescale,
                crop=crop,
                rotate=rotate,
                flip=flip
            )

            transform = np.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation

            new_imgs.append(self.color_normalize(new_img))
            transforms.append(transform)

        results['img'] = new_imgs
        results['img_aug_matrix'] = np.stack(transforms)
        return results


@PIPELINES.register_module(force=True)
class LoadAnnotations3D(_LoadAnnotations3D):

    def __init__(
        self, pts_range: Tuple = None,
        bev_grid_size: Tuple = None, class_names: Tuple = None,
        with_bbox_3d: bool = False, with_label_3d: bool = False,
        with_seg: bool = False, **kwargs):
        super(LoadAnnotations3D, self).__init__(
            with_bbox_3d=with_bbox_3d,
            with_label_3d=with_label_3d,
            with_seg=with_seg, **kwargs)
        self.pts_range = pts_range
        self.bev_grid_size = bev_grid_size
        self.class_names = class_names

    def _load_semantic_seg(self, results):
        instance_masks = rasterize_map(results['ann_info']['lane_polygons'],
                                       (self.pts_range[4] - self.pts_range[1],
                                        self.pts_range[3] - self.pts_range[0]),
                                       self.bev_grid_size,
                                       lane_types=self.class_names,
                                       thickness=5)
        semantic_masks = np.vstack([
            ~np.any(instance_masks, axis=0, keepdims=True), instance_masks != 0])

        results['gt_semantic_seg'] = semantic_masks.astype(float) # onehot
        results['seg_fields'].append('gt_semantic_seg')
        return results
