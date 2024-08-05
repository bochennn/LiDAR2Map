from copy import deepcopy
from typing import Dict, List

import torch
from mmcv.runner import BaseModule, force_fp32
from mmdet3d.core import circle_nms
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
# from mmdet.core import multi_apply

from .center_head import CenterHead


@HEADS.register_module()
class MultiScaleCenterHead(BaseModule):

    def __init__(
        self,
        in_channels: List[int] = None,
        tasks: List[Dict] = None,
        out_size_factor: List[int] = None,
        class_names: List[str] = None,
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='none', loss_weight=0.25),
        **kwargs
    ):
        super(MultiScaleCenterHead, self).__init__()
        assert len(in_channels) == len(tasks) == len(out_size_factor)

        self.tasks = tasks
        self.class_names = class_names
        self.code_size = kwargs['bbox_coder']['code_size']
        self.test_cfg = kwargs['test_cfg']

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.with_velocity = 'vel' in kwargs['common_heads'].keys()

        for i, _task in enumerate(tasks):
            kwargs['bbox_coder'].update(out_size_factor=out_size_factor[i])
            kwargs['train_cfg'].update(out_size_factor=out_size_factor[i])
            setattr(self, f'head_{i}', CenterHead(in_channels=in_channels[i],
                                                  tasks=_task, **deepcopy(kwargs)))

    def forward(self, lvl_feats: List[torch.Tensor]):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        ret_dicts = []
        for i in range(len(self.tasks)):
            x = getattr(self, f'head_{i}').shared_conv(lvl_feats[i])
            ret_dicts.append([task(x) for task in getattr(self, f'head_{i}').task_heads])
        return ret_dicts

    def loss_single(self, heatmaps: List[torch.Tensor], anno_boxes, inds, masks,
                    preds_dict: List[Dict], head_id: int, task_id: int):

        preds_dict[task_id]['heatmap'] = clip_sigmoid(preds_dict[task_id]['heatmap'])
        num_pos = heatmaps[task_id].eq(1).float().sum().item()

        loss_heatmap = self.loss_cls(
            preds_dict[task_id]['heatmap'],
            heatmaps[task_id],
            avg_factor=max(num_pos, 1))
        target_box = anno_boxes[task_id]
        # reconstruct the anno_box from multiple reg heads
        if self.with_velocity:
            preds_dict[task_id]['anno_box'] = torch.cat(
                (preds_dict[task_id]['reg'], preds_dict[task_id]['height'],
                 preds_dict[task_id]['dim'], preds_dict[task_id]['rot'],
                 preds_dict[task_id]['vel']),
                dim=1)
        else:
            preds_dict[task_id]['anno_box'] = torch.cat(
                (preds_dict[task_id]['reg'], preds_dict[task_id]['height'],
                 preds_dict[task_id]['dim'], preds_dict[task_id]['rot']),
                dim=1)
        # Regression loss for dimension, offset, height, rotation
        ind = inds[task_id]
        num = masks[task_id].float().sum()
        pred = preds_dict[task_id]['anno_box'].permute(0, 2, 3, 1).contiguous()
        pred = pred.view(pred.size(0), -1, pred.size(3))
        pred = getattr(self, f'head_{head_id}')._gather_feat(pred, ind)
        mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
        isnotnan = (~torch.isnan(target_box)).float()
        mask *= isnotnan

        code_weights = getattr(self, f'head_{head_id}').train_cfg.get('code_weights', None)
        bbox_weights = mask * mask.new_tensor(code_weights)
        loss_bbox = self.loss_bbox(
            pred, target_box, bbox_weights, avg_factor=(num + 1e-4))

        return {
            f'head{head_id}_task{task_id}.loss_heatmap': loss_heatmap,
            f'head{head_id}_task{task_id}.loss_bbox': loss_bbox
        }

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs) -> Dict:
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        loss_dict = dict()
        for head_id, preds_dict in enumerate(preds_dicts):
            heatmaps, anno_boxes, inds, masks = getattr(
                self, f'head_{head_id}').get_targets(gt_bboxes_3d, gt_labels_3d)

            for task_id in range(len(preds_dict)):
                loss_dict.update(self.loss_single(
                    heatmaps, anno_boxes, inds, masks, preds_dict, head_id, task_id))

        return loss_dict

    def get_bboxes_single(self, preds_dict, num_class_with_bg, img_metas,
                          head_id: int, task_id: int):

        batch_size = preds_dict[task_id]['heatmap'].shape[0]
        batch_heatmap = preds_dict[task_id]['heatmap'].sigmoid()

        batch_reg = preds_dict[task_id]['reg']
        batch_hei = preds_dict[task_id]['height']

        if getattr(self, f'head_{head_id}').norm_bbox:
            batch_dim = torch.exp(preds_dict[task_id]['dim'])
        else:
            batch_dim = preds_dict[task_id]['dim']

        batch_rots = preds_dict[task_id]['rot'][:, 0].unsqueeze(1)
        batch_rotc = preds_dict[task_id]['rot'][:, 1].unsqueeze(1)

        batch_vel = preds_dict[task_id]['vel'] if 'vel' in preds_dict[task_id] else None

        temp = getattr(self, f'head_{head_id}').bbox_coder.decode(
            batch_heatmap,
            batch_rots,
            batch_rotc,
            batch_hei,
            batch_dim,
            batch_vel,
            reg=batch_reg,
            task_id=task_id)
        assert self.test_cfg['nms_type'] in ['circle', 'rotate']
        batch_reg_preds = [box['bboxes'] for box in temp]
        batch_cls_preds = [box['scores'] for box in temp]
        batch_cls_labels = [box['labels'] for box in temp]

        ret_task = []
        if self.test_cfg['nms_type'] == 'circle':
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
            ret_task = getattr(self, f'head_{head_id}').get_task_detections(
                    num_class_with_bg[task_id], batch_cls_preds, batch_reg_preds,
                    batch_cls_labels, img_metas)

        for batch_ret in ret_task:
            batch_ret.update(labels=[getattr(
                self, f'head_{head_id}').class_names[task_id][label]
                for label in batch_ret['labels']])

        return ret_task # B[dict]

    def get_bboxes(self, preds_dicts: List, img_metas: List, **kwargs: Dict) -> List:
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets, ret_list = [], [] # [tasks, B, dict]
        for head_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = getattr(self, f'head_{head_id}').num_classes

            for task_id in range(len(preds_dict)):
                rets.append(self.get_bboxes_single(
                    preds_dict, num_class_with_bg, img_metas, head_id, task_id))

        # Merge branches results
        batch_size = len(rets[0])
        for i in range(batch_size):
            for k in rets[0][0].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](bboxes, self.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    labels = scores.new_tensor([
                        self.class_names.index(cn) for ret in rets for cn in ret[i][k]], dtype=int)
            ret_list.append([bboxes, scores, labels])
        return ret_list
