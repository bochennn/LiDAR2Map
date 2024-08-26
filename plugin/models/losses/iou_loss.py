import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.diff_iou_rotated import box2corners, oriented_box_intersection_2d
from mmdet3d.models.builder import LOSSES

__all__ = ['IoU3DLoss']


def diff_iou_rotated_3d(box3d1: torch.Tensor, box3d2: torch.Tensor, mode: str = 'iou') -> torch.Tensor:
    """Calculate differentiable iou of rotated 3d boxes.

    Args:
        box3d1 (Tensor): (B, N, 3+3+1) First box (x,y,z,w,h,l,alpha).
        box3d2 (Tensor): (B, N, 3+3+1) Second box (x,y,z,w,h,l,alpha).

    Returns:
        Tensor: (B, N) IoU.
    """
    box1 = box3d1[..., [0, 1, 3, 4, 6]]  # 2d box
    box2 = box3d2[..., [0, 1, 3, 4, 6]]
    corners1 = box2corners(box1)
    corners2 = box2corners(box2)
    intersection, _ = oriented_box_intersection_2d(corners1, corners2)
    zmax1 = box3d1[..., 2] + box3d1[..., 5] * 0.5
    zmin1 = box3d1[..., 2] - box3d1[..., 5] * 0.5
    zmax2 = box3d2[..., 2] + box3d2[..., 5] * 0.5
    zmin2 = box3d2[..., 2] - box3d2[..., 5] * 0.5
    z_overlap = (torch.min(zmax1, zmax2) -
                 torch.max(zmin1, zmin2)).clamp_(min=0.)
    intersection_3d = intersection * z_overlap
    volume1 = box3d1[..., 3] * box3d1[..., 4] * box3d1[..., 5]
    volume2 = box3d2[..., 3] * box3d2[..., 4] * box3d2[..., 5]
    union_3d = volume1 + volume2 - intersection_3d

    if mode == 'iou':
        return intersection_3d / union_3d

    elif mode == 'diou':
        inter_diag = torch.pow(box3d1[..., 0:3] - box3d2[..., 0:3], 2).sum(-1)

        outer_h = torch.maximum(box3d1[..., 2] + 0.5 * box3d1[..., 5], box3d2[..., 2] + 0.5 * box3d2[..., 5]) - \
                  torch.minimum(box3d1[..., 2] - 0.5 * box3d1[..., 5], box3d2[..., 2] - 0.5 * box3d2[..., 5])
        outer_h = torch.clamp(outer_h, min=0)

        out_max_xy = torch.maximum(corners1[:, :, 2], corners2[:, :, 2])
        out_min_xy = torch.minimum(corners1[:, :, 0], corners2[:, :, 0])
        outer = torch.clamp((out_max_xy - out_min_xy), min=0)
        outer_diag = outer[..., 0] ** 2 + outer[..., 1] ** 2 + outer_h ** 2

        dious = intersection_3d / union_3d - inter_diag / outer_diag
        return torch.clamp(dious, min=-1.0, max=1.0)
    else:
        raise NotImplementedError


@LOSSES.register_module()
class IoU3DLoss(nn.Module):

    def __init__(self, reg_weight=1.0):
        super().__init__()
        self.reg_weight = reg_weight

    def forward(self, pred_iou: torch.Tensor, pred_boxes: torch.Tensor,
                gt_boxes: torch.Tensor, mask: torch.Tensor):
        if mask.sum() == 0:
            return (pred_iou * 0.).sum()
        mask = mask.bool()

        selected_pred_boxes = torch.cat([
            _box['bboxes'][_mask] for _box, _mask in zip(pred_boxes, mask)])
        selected_pred_boxes = selected_pred_boxes.unsqueeze(0)

        selected_gt_boxes = gt_boxes[mask]
        selected_gt_boxes[:, 6:7] = torch.atan2(selected_gt_boxes[:, 6:7],
                                                selected_gt_boxes[:, 7:8])
        selected_gt_boxes = selected_gt_boxes[:, :7].unsqueeze(0)

        target_iou = diff_iou_rotated_3d(selected_pred_boxes, selected_gt_boxes).view(-1)
        target_iou = target_iou * 2 - 1  # [0, 1] ==> [-1, 1]

        iou_loss = F.l1_loss(pred_iou[mask].view(-1), target_iou, reduction='sum')
        iou_loss = iou_loss / torch.clamp(mask.sum(), min=1e-4)

        diou = diff_iou_rotated_3d(selected_pred_boxes, selected_gt_boxes, mode='diou').view(-1)
        iou_reg_loss = (1.0 - diou).sum() / torch.clamp(mask.sum(), min=1e-4)

        return iou_loss + iou_reg_loss * self.reg_weight
