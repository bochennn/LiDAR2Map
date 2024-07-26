import torch
from .lovasz_softmax_loss import lovasz_softmax
import torch.nn.functional as F

__all__ = ['logits_loss']


def logits_loss(pd_logits: torch.Tensor, gt_onehot: torch.Tensor):
    semantic_labels = gt_onehot.argmax(dim=1)
    pd_classes = F.softmax(pd_logits, dim=1)

    bce_loss = F.binary_cross_entropy_with_logits(pd_logits, gt_onehot)
    lovaz_loss = lovasz_softmax(pd_classes, semantic_labels, ignore=0)
    return bce_loss + lovaz_loss
