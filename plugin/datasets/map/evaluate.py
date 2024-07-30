import numpy as np
import torch


def onehot_iou_torch(pd_logits: torch.Tensor, gt_logits: torch.Tensor):
    """
    semantic mask: B, cls, H, W
    cls[0] is background
    """
    pd_logits = pd_logits[:, 1:].flatten(2).bool()
    gt_logits = gt_logits[:, 1:].flatten(2).bool()

    intersects = (pd_logits & gt_logits).sum(dim=2).float()
    unions = (pd_logits | gt_logits).sum(dim=2).float()
    return intersects / (unions + 1e-7)


def onehot_iou_numpy(pd_logits: np.ndarray, gt_logits: np.ndarray):
    """
    semantic mask: cls, H, W
    cls[0] is background
    """
    pd_logits = pd_logits[1:].reshape(pd_logits.shape[0], -1).astype(bool)
    gt_logits = gt_logits[1:].reshape(gt_logits.shape[0], -1).astype(bool)

    intersects = (pd_logits & gt_logits).sum(axis=1).astype(float)
    unions = (pd_logits | gt_logits).sum(axis=1).astype(float)
    return intersects / (unions + 1e-7)
