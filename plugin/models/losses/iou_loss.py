import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import diff_iou_rotated_3d
from mmdet3d.models.builder import LOSSES

__all__ = ['IoU3DLoss']


@LOSSES.register_module()
class IoU3DLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred_iou: torch.Tensor, pred_boxes: torch.Tensor,
                gt_boxes: torch.Tensor, mask: torch.Tensor):
        if mask.sum() == 0:
            return pred_iou.new_zeros((1))
        mask = mask.bool()

        selected_pred_boxes = torch.cat([_box['bboxes'] for _box in pred_boxes])

        selected_gt_boxes = gt_boxes[mask]
        selected_gt_boxes[:, 6:7] = torch.atan2(selected_gt_boxes[:, 6:7],
                                                selected_gt_boxes[:, 7:8])
        target_iou = diff_iou_rotated_3d(selected_pred_boxes.unsqueeze(0),
                                         selected_gt_boxes[None, :, :7]).view(-1)
        # iou = box_utils.bbox3d_overlaps_diou()

        target_iou = target_iou * 2 - 1  # [0, 1] ==> [-1, 1]

        loss = F.l1_loss(pred_iou[mask].view(-1), target_iou, reduction='sum')
        loss = loss / torch.clamp(mask.sum(), min=1e-4)

        return loss * self.loss_weight
