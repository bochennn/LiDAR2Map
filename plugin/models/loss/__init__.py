import torch.nn.functional as F

from .tree_filter.modules.tree_filter import MinimumSpanningTree, TreeFilter2D


def compute_feature_distill_loss(features_preds, features_targets, low_feats):
    weight = 0.4
    mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
    tree_filter_layers = TreeFilter2D(groups=1, sigma=0.02)
    feature_distill_loss = 0.0
    resize_shape = features_preds[-4].shape[-2:]
    if isinstance(features_preds, list):
        for i in range(len(features_preds)):  # 1/8   1/16   1/32
            feature_target = features_targets[i].detach()
            feature_pred = features_preds[i]

            B, C, H, W = feature_pred.shape
            feature_pred_down = F.interpolate(feature_pred, size=resize_shape, mode="bilinear")
            feature_target_down = F.interpolate(feature_target, size=resize_shape, mode="bilinear")
            low_feats = F.interpolate(low_feats, size=resize_shape, mode="bilinear")

            tree = mst_layers(low_feats)
            fusion_affinity = tree_filter_layers(feature_in=feature_target_down,
                                                 embed_in=low_feats, tree=tree, low_tree=False)

            tree = mst_layers(feature_target_down)
            fusion_affinity = tree_filter_layers(feature_in=fusion_affinity,
                                                 embed_in=feature_target_down, tree=tree, low_tree=False)

            tree = mst_layers(low_feats)
            lidar_affinity = tree_filter_layers(feature_in=feature_pred_down,
                                                embed_in=low_feats, tree=tree, low_tree=False)

            tree = mst_layers(feature_target_down)
            lidar_affinity = tree_filter_layers(feature_in=lidar_affinity,
                                                embed_in=feature_pred_down, tree=tree, low_tree=False)

            feature_distill_loss = feature_distill_loss + F.l1_loss(lidar_affinity, fusion_affinity, reduction='mean') / B

    else:
        feature_target = features_targets.detach()
        feature_pred = features_preds

        B, C, H, W = feature_pred.shape
        feature_pred_down = F.interpolate(feature_pred, size=resize_shape, mode="bilinear", align_corners=False)
        feature_target_down = F.interpolate(feature_target, size=resize_shape, mode="bilinear", align_corners=False)
        low_feats = F.interpolate(low_feats, size=resize_shape, mode="bilinear")

        tree = mst_layers(low_feats)
        fusion_affinity = tree_filter_layers(feature_in=feature_target_down,
                                             embed_in=low_feats, tree=tree)

        tree = mst_layers(feature_target_down)
        fusion_affinity = tree_filter_layers(feature_in=fusion_affinity,
                                             embed_in=feature_target_down, tree=tree, low_tree=False)

        tree = mst_layers(low_feats)
        lidar_affinity = tree_filter_layers(feature_in=feature_pred_down,
                                            embed_in=low_feats, tree=tree)

        tree = mst_layers(feature_target_down)
        lidar_affinity = tree_filter_layers(feature_in=lidar_affinity,
                                            embed_in=feature_pred_down, tree=tree, low_tree=False)

        feature_distill_loss = feature_distill_loss + F.l1_loss(lidar_affinity, fusion_affinity, reduction='mean') / B

    return weight * feature_distill_loss


def compute_logit_distill_loss(logits_preds, logits_targets):
    weight = 1.5
    logit_distill_loss = 0.0
    if isinstance(logits_preds, list):
        for i in range(len(logits_preds)):
            preds_temp = logits_preds[i]
            targets_temp = logits_targets[i]
            logit_distill_loss = logit_distill_loss + weight * F.kl_div(F.log_softmax(preds_temp, dim=1),
                                                                        F.softmax(targets_temp.detach(), dim=1),
                                                                        reduction='none').sum(1).mean()
    else:
        logit_distill_loss = weight * F.kl_div(F.log_softmax(logits_preds, dim=1),
                                               F.softmax(logits_targets.detach(), dim=1),
                                               reduction='none').sum(1).mean()
    return logit_distill_loss
