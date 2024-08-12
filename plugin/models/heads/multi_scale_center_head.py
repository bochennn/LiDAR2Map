from copy import deepcopy
from typing import Dict, List

import torch
from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.builder import HEADS, build_loss

from .center_head import CenterHead


@HEADS.register_module()
class MultiScaleCenterHead(BaseModule):

    def __init__(
        self,
        in_channels: List[int] = None,
        tasks: List[Dict] = None,
        out_size_factor: List[int] = None,
        class_names: List[str] = None,
        loss_cls: Dict = dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox: Dict = dict(type='L1Loss', reduction='none', loss_weight=0.25),
        **kwargs
    ):
        super(MultiScaleCenterHead, self).__init__()
        assert len(in_channels) == len(tasks) == len(out_size_factor)

        self.tasks = tasks
        self.class_names = class_names
        self.code_size = kwargs['bbox_coder']['code_size']
        self.test_cfg = kwargs.get('test_cfg')

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.with_velocity = 'vel' in kwargs['common_heads'].keys()

        for i, sub_tasks in enumerate(tasks):
            kwargs['bbox_coder'].update(out_size_factor=out_size_factor[i])

            if kwargs.get('train_cfg') is not None:
                kwargs['train_cfg'].update(out_size_factor=out_size_factor[i])
            if kwargs.get('test_cfg') is not None:
                kwargs['test_cfg'].update(out_size_factor=out_size_factor[i])

            setattr(self, f'head_{i}', CenterHead(in_channels=in_channels[i],
                                                  class_names=class_names,
                                                  tasks=sub_tasks,
                                                  **deepcopy(kwargs)))

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
            single_head: CenterHead = getattr(self, f'head_{i}')
            feats = single_head.shared_conv(lvl_feats[i])
            ret_dicts.append([task(feats) for task in single_head.task_heads])
        return ret_dicts

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
            heatmaps, anno_boxes, inds, masks = getattr(self, f'head_{head_id}').\
                get_targets(gt_bboxes_3d, gt_labels_3d)

            for task_id, preds_with_targets in enumerate(
                    zip(preds_dict, heatmaps, anno_boxes, inds, masks)):
                loss_heatmap, loss_bbox, loss_iou = getattr(self, f'head_{head_id}').\
                    loss_single(*preds_with_targets)
                loss_dict[f'head{head_id}_task{task_id}.loss_heatmap'] = loss_heatmap
                loss_dict[f'head{head_id}_task{task_id}.loss_bbox'] = loss_bbox
                if loss_iou is not None:
                    loss_dict[f'head{head_id}_task{task_id}.loss_iou'] = loss_iou
        return loss_dict

    @torch.no_grad()
    def get_bboxes(self, preds_dicts: List, img_metas: List, **kwargs: Dict) -> List:
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        task_rets = [] # [tasks, B, dict]
        for head_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = getattr(self, f'head_{head_id}').num_classes

            for task_id, task_preds in enumerate(zip(preds_dict, num_class_with_bg)):
                task_rets.append(getattr(self, f'head_{head_id}').\
                            get_bboxes_single(*task_preds, img_metas, task_id))

        # Merge branches results
        return getattr(self, f'head_0')._merge_task_results(task_rets, img_metas)
