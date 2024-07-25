from typing import List
import torch
import torch.nn.functional as F

from .tree_filter.modules.tree_filter import MinimumSpanningTree, TreeFilter2D
from .loss import SimpleLoss


def feature_distill_loss(
    features_preds: List[torch.Tensor],
    features_targets: List[torch.Tensor],
    low_feats: torch.Tensor
):
    """
    features_preds: student feature from lidar backbone
    features_targets: teacher feature from fusion backbone
    """
    mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
    tree_filter_layers = TreeFilter2D(groups=1, sigma=0.02)
    feature_distill_loss = 0.0
    resize_shape = features_preds[-4].shape[-2:]
    B = low_feats.shape[0]

    for i in range(len(features_preds)):  # 1/8   1/16   1/32
        feature_target = features_targets[i].detach()
        feature_pred = features_preds[i]

        feature_pred_down = F.interpolate(feature_pred, size=resize_shape, mode="bilinear")
        feature_target_down = F.interpolate(feature_target, size=resize_shape, mode="bilinear")
        low_feats = F.interpolate(low_feats, size=resize_shape, mode="bilinear")

        fusion_affinity = tree_filter_layers(feature_in=feature_target_down, embed_in=low_feats,
                                             tree=mst_layers(low_feats), low_tree=False)
        fusion_affinity = tree_filter_layers(feature_in=fusion_affinity, embed_in=feature_target_down,
                                             tree=mst_layers(feature_target_down), low_tree=False)

        lidar_affinity = tree_filter_layers(feature_in=feature_pred_down, embed_in=low_feats,
                                            tree=mst_layers(low_feats), low_tree=False)
        lidar_affinity = tree_filter_layers(feature_in=lidar_affinity, embed_in=feature_pred_down,
                                            tree=mst_layers(feature_target_down), low_tree=False)

        feature_distill_loss += F.l1_loss(lidar_affinity, fusion_affinity, reduction='mean') / B

    return feature_distill_loss


def logit_distill_loss(logits_preds: torch.Tensor, logits_targets: torch.Tensor):
    logit_distill_loss = F.kl_div(F.log_softmax(logits_preds, dim=1),
                                    F.softmax(logits_targets.detach(), dim=1),
                                    reduction='none').sum(dim=1).mean()
    return logit_distill_loss
