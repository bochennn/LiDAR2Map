from typing import Dict, List

import torch
from mmcv.runner import force_fp32
from mmdet3d.core import circle_nms, draw_heatmap_gaussian, gaussian_radius
from mmdet3d.core.bbox.coders.centerpoint_bbox_coders import \
    CenterPointBBoxCoder as _CenterPointBBoxCoder
from mmdet3d.models.builder import HEADS
from mmdet3d.models.dense_heads.centerpoint_head import \
    CenterHead as _CenterHead
from mmdet3d.models.utils import clip_sigmoid
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module(force=True)
class CenterPointBBoxCoder(_CenterPointBBoxCoder):

    def decode(self,
               heat: torch.Tensor,
               rot_sine: torch.Tensor,
               rot_cosine: torch.Tensor,
               hei: torch.Tensor,
               dim: torch.Tensor,
               vel: torch.Tensor,
               reg: torch.Tensor = None,
               task_id: int = -1):
        """Decode bboxes.

        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].
            rot_sine (torch.Tensor): Sine of rotation with the shape of
                [B, 1, W, H].
            rot_cosine (torch.Tensor): Cosine of rotation with the shape of
                [B, 1, W, H].
            hei (torch.Tensor): Height of the boxes with the shape
                of [B, 1, W, H].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 1, W, H].
            vel (torch.Tensor): Velocity with the shape of [B, 1, W, H].
            reg (torch.Tensor, optional): Regression value of the boxes in
                2D with the shape of [B, 2, W, H]. Default: None.
            task_id (int, optional): Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.
        """
        batch, _, _, _ = heat.size()

        scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)

        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, self.max_num, 2)
            xs = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, self.max_num, 1) + 0.5
            ys = ys.view(batch, self.max_num, 1) + 0.5

        # rotation value and direction label
        rot_sine = self._transpose_and_gather_feat(rot_sine, inds)
        rot_sine = rot_sine.view(batch, self.max_num, 1)

        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)
        rot_cosine = rot_cosine.view(batch, self.max_num, 1)
        rot = torch.atan2(rot_sine, rot_cosine)

        # height in the bev
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, self.max_num, 1)

        # dim of the box
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, self.max_num, 3)

        # class label
        clses = clses.view(batch, self.max_num).float()
        scores = scores.view(batch, self.max_num)

        xs = xs.view(batch, self.max_num, 1) * \
            self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        ys = ys.view(batch, self.max_num, 1) * \
            self.out_size_factor * self.voxel_size[1] + self.pc_range[1]

        if vel is None:  # KITTI FORMAT
            final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)
        else:  # exist velocity, nuscene format
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, self.max_num, 2)
            final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2)

        final_scores = scores
        final_preds = clses

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:

            if isinstance(self.post_center_range, list):
                self.post_center_range = torch.tensor(
                    self.post_center_range, device=heat.device)

            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts


@HEADS.register_module(force=True)
class CenterHead(_CenterHead):

    def __init__(self, class_names: List = None, tasks: List[Dict] = None, **kwargs):
        """ """
        for task in tasks:
            task.update(class_names=[class_names.index(c) for c in task['class_names']])
        super(CenterHead, self).__init__(tasks=tasks, **kwargs)

        # self.iou_loss = build_loss(dict(type='IoU3DLoss'))

    def get_targets_single(
        self,
        gt_bboxes_3d,
        gt_labels_3d: torch.Tensor
    ):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes
                    are valid.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = torch.div(grid_size[:2], self.train_cfg['out_size_factor'], rounding_mode='trunc')

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names: # label id
            # task_masks.append([
            #     torch.where(gt_labels_3d == class_name.index(i) + flag)
            #     for i in class_name
            # ])
            task_masks.append([
                torch.where(gt_labels_3d == label_id) for label_id in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(torch.tensor(
                    [self.class_names[idx].index(i) + 1 for i in gt_labels_3d[m]]))
                # task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1], feature_map_size[0]))

            if self.with_velocity:
                anno_box = gt_bboxes_3d.new_zeros((max_objs, 10), dtype=torch.float32)
            else:
                anno_box = gt_bboxes_3d.new_zeros((max_objs, 8), dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg['out_size_factor']
                length = length / voxel_size[1] / self.train_cfg['out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][1], task_boxes[idx][k][2]

                    coor_x = (x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                        and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x < feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    if self.with_velocity:
                        vx, vy = task_boxes[idx][k][7:]
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device=device),
                            z.unsqueeze(0), box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0),
                            vx.unsqueeze(0),
                            vy.unsqueeze(0)
                        ])
                    else:
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device=device),
                            z.unsqueeze(0), box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0)
                        ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    def loss_single(
        self,
        task_preds: Dict[str, torch.Tensor],
        heatmap: torch.Tensor,
        target_box: torch.Tensor,
        ind: torch.Tensor,
        mask: torch.Tensor,
    ):
        if isinstance(task_preds, list):
            task_preds = task_preds[0]

        task_preds['heatmap'] = clip_sigmoid(task_preds['heatmap'])
        num_pos = heatmap.eq(1).float().sum().item()

        loss_heatmap = self.loss_cls(task_preds['heatmap'], heatmap,
                                     avg_factor=max(num_pos, 1))

        # reconstruct the anno_box from multiple reg heads
        if self.with_velocity:
            task_preds['anno_box'] = torch.cat(
                (task_preds['reg'], task_preds['height'],
                 task_preds['dim'], task_preds['rot'],
                 task_preds['vel']), dim=1)
        else:
            task_preds['anno_box'] = torch.cat(
                (task_preds['reg'], task_preds['height'],
                 task_preds['dim'], task_preds['rot']), dim=1)
        # Regression loss for dimension, offset, height, rotation
        num = mask.float().sum()
        # pred = task_preds['anno_box'].permute(0, 2, 3, 1).contiguous()
        # pred = pred.view(pred.size(0), -1, pred.size(3))
        pred = self.bbox_coder._transpose_and_gather_feat(task_preds['anno_box'], ind)
        isnotnan = (~torch.isnan(target_box)).float()
        code_mask = mask.unsqueeze(2).expand_as(target_box).float() * isnotnan

        code_weights = self.train_cfg.get('code_weights', None)
        bbox_weights = code_mask * code_mask.new_tensor(code_weights)
        loss_bbox = self.loss_bbox(
            pred, target_box, bbox_weights, avg_factor=(num + 1e-4))

        if 'iou' in task_preds:
            # with torch
            target_box_preds = self.bbox_coder.decode(
                heatmap.eq(1).float(),
                task_preds['rot'][:, 0:1], task_preds['rot'][:, 1:2],
                task_preds['height'], task_preds['dim'],
                task_preds.get('vel'), task_preds['reg']
            )  # (B, H, W, 7 or 9)

            pred_iou = self.bbox_coder._transpose_and_gather_feat(task_preds['iou'], ind)
            iou_loss = self.iou_loss(pred_iou, target_box_preds, target_box, mask)
        else:
            iou_loss = None

        return loss_heatmap, loss_bbox, iou_loss

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        loss_dict = dict()
        for task_id, preds_with_targets in enumerate(
                zip(preds_dicts, heatmaps, anno_boxes, inds, masks)):
            loss_heatmap, loss_bbox, loss_iou = self.loss_single(*preds_with_targets)
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
            if loss_iou is not None:
                loss_dict[f'task{task_id}.loss_iou'] = loss_iou

        return loss_dict

    def get_bboxes_single(
        self,
        task_preds: Dict[str, torch.Tensor],
        num_class_with_bg: int,
        img_metas: List,
        task_id: int
    ) -> List[Dict]:
        batch_size = task_preds['heatmap'].shape[0]
        batch_heatmap = task_preds['heatmap'].sigmoid()

        batch_reg = task_preds['reg']
        batch_hei = task_preds['height']

        batch_dim = torch.exp(task_preds['dim']) if self.norm_bbox else task_preds['dim']

        batch_rots = task_preds['rot'][:, 0].unsqueeze(1)
        batch_rotc = task_preds['rot'][:, 1].unsqueeze(1)
        batch_vel = task_preds.get('vel')

        temp = self.bbox_coder.decode(batch_heatmap, batch_rots, batch_rotc,
                                      batch_hei, batch_dim, batch_vel,
                                      reg=batch_reg, task_id=task_id)
        assert self.test_cfg['nms_type'] in ['circle', 'rotate']
        batch_reg_preds = [box['bboxes'] for box in temp]
        batch_cls_preds = [box['scores'] for box in temp]
        batch_cls_labels = [box['labels'] for box in temp]

        if self.test_cfg['nms_type'] == 'circle':
            ret_task = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                centers = boxes3d[:, [0, 1]]
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                keep = torch.tensor(
                    circle_nms(
                        boxes.detach().cpu().numpy(),
                        self.test_cfg['min_radius'][task_id],
                        post_max_size=self.test_cfg['post_max_size']),
                    dtype=torch.long,
                    device=boxes.device)

                boxes3d = boxes3d[keep]
                scores = scores[keep]
                labels = labels[keep]
                ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_task.append(ret)
        else:
            ret_task = self.get_task_detections(
                    num_class_with_bg, batch_cls_preds, batch_reg_preds,
                    batch_cls_labels, img_metas)

        for batch_ret in ret_task:
            batch_labels = batch_ret['labels']
            batch_ret.update(
                labels=batch_labels.new_tensor([self.class_names[task_id][label]
                for label in batch_labels]))

        return ret_task # B[dict]

    def _merge_task_results(self, task_results: List, img_metas: List):
        """ task_results: [tasks, B, dict] -> [B, dict] """
        batch_size, ret_list = len(task_results[0]), []
        for i in range(batch_size):
            for k in task_results[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in task_results])
                    # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size, origin=(0.5, 0.5, 0.5))
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in task_results])
                elif k == 'labels':
                    labels = torch.cat([ret[i][k].int() for ret in task_results])
            ret_list.append([bboxes, scores, labels])
        return ret_list

    @torch.no_grad()
    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        task_rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            task_rets.append(self.get_bboxes_single(
                preds_dict[0],
                self.num_classes[task_id],
                img_metas, task_id))

        # Merge branches results
        return self._merge_task_results(task_rets, img_metas)
