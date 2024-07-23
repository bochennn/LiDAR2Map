from typing import Any, Dict, Tuple

import numpy as np
from mmdet3d.datasets.builder import PIPELINES
from PIL import Image


@PIPELINES.register_module()
class PrepareImageInputs(object):

    def __init__(
        self,
        pre_scale: Tuple[float, float],
        pre_crop: Tuple[int, int],
        rand_scale: Tuple[float, float] = (1.0, 1.0),
        rand_rotation: Tuple[float, float] = (0.0, 0.0),
        rand_flip: bool = False
    ):
        self.pre_scale = pre_scale
        self.pre_crop = pre_crop
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
            new_imgs.append(np.array(new_img).astype(np.float32))
            transforms.append(transform)

        results['img'] = new_imgs
        results['img_aug_matrix'] = np.stack(transforms)
        return results
