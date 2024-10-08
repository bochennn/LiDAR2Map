import torch
import torch.nn as nn
import torch_scatter
from mmdet3d.models.voxel_encoders.pillar_encoder import DynamicPillarFeatureNet as _DynamicPillarFeatureNet
from mmdet3d.models.builder import VOXEL_ENCODERS

from .voxel import points_to_voxels


@VOXEL_ENCODERS.register_module(force=True)
class DynamicPillarFeatureNet(_DynamicPillarFeatureNet):
    def forward(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        batch_size: int,
        **kwargs
    ):
        voxel_feats, voxel_coors = super(
            DynamicPillarFeatureNet, self).forward(features=voxels, coors=coors)

        return dict(
            batch_size=batch_size,
            voxel_features=voxel_feats,
            coors=voxel_coors
        )

class PillarBlock(nn.Module):
    def __init__(self, idims=64, dims=64, num_layers=1,
                stride=1):
        super(PillarBlock, self).__init__()
        layers = []
        self.idims = idims
        self.stride = stride
        for i in range(num_layers):
            layers.append(nn.Conv2d(self.idims, dims, 3, stride=self.stride,
                                    padding=1, bias=False))
            layers.append(nn.BatchNorm2d(dims))
            layers.append(nn.ReLU(inplace=True))
            self.idims = dims
            self.stride = 1
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PointNet(nn.Module):
    def __init__(self, idims=64, odims=64):
        super(PointNet, self).__init__()
        self.pointnet = nn.Sequential(
            nn.Conv1d(idims, odims, kernel_size=1, bias=False),
            nn.BatchNorm1d(odims),
            nn.ReLU(inplace=True)
        )

    def forward(self, points_feature, points_mask):
        batch_size, num_points, num_dims = points_feature.shape
        points_feature = points_feature.permute(0, 2, 1)
        mask = points_mask.view(batch_size, 1, num_points)
        return self.pointnet(points_feature) * mask


def convert_relu_to_softplus(model, act):
    for child_name, child in model.named_children():
        if isinstance(child, nn.LeakyReLU):
            setattr(model, child_name, act)
        else:
            convert_relu_to_softplus(child, act)


# class PointPillar(nn.Module):
#   def __init__(self, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36):
#     super(PointPillar, self).__init__()
#     self.xbound = data_conf['xbound']
#     self.ybound = data_conf['ybound']
#     self.zbound = data_conf['zbound']
#     self.embedded_dim = embedded_dim
#     self.pn = PointNet(15, 64)
#     self.block1 = PillarBlock(64, dims=64, num_layers=2, stride=1)
#     self.block2 = PillarBlock(64, dims=128, num_layers=3, stride=2)
#     self.block3 = PillarBlock(128, 256, num_layers=3, stride=2)
#     self.up1 = nn.Sequential(
#       nn.Conv2d(64, 64, 3, padding=1, bias=False),
#       nn.BatchNorm2d(64),
#       nn.ReLU(inplace=True)
#     )
#     self.up2 = nn.Sequential(
#       nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#       nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
#       nn.BatchNorm2d(128),
#       nn.ReLU(inplace=True)
#     )
#     self.up3 = nn.Sequential(
#       nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
#       nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
#       nn.BatchNorm2d(256),
#       nn.ReLU(inplace=True)
#     )
#     convert_relu_to_softplus(self.block1, nn.Hardswish())
#     convert_relu_to_softplus(self.block2, nn.Hardswish())
#     convert_relu_to_softplus(self.block3, nn.Hardswish())
#     convert_relu_to_softplus(self.up1, nn.Hardswish())
#     convert_relu_to_softplus(self.up2, nn.Hardswish())
#     convert_relu_to_softplus(self.up3, nn.Hardswish())
#     # self.dropout_lidar = nn.Dropout2d(p=0.2)
#     self.conv_out = nn.Sequential(
#       nn.Conv2d(448, 256, 3, padding=1, bias=False),
#       nn.BatchNorm2d(256),
#       nn.ReLU(inplace=True),
#       nn.Conv2d(256, 128, 3, padding=1, bias=False),
#       nn.BatchNorm2d(128),
#       nn.ReLU(inplace=True),
#       nn.Conv2d(128, data_conf['num_channels'], 1),
#     )

#     if instance_seg:
#       self.instance_conv_out = nn.Sequential(
#         nn.Conv2d(448, 256, 3, padding=1, bias=False),
#         nn.BatchNorm2d(256),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(256, 128, 3, padding=1, bias=False),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(128, embedded_dim, 1),
#       )
#     if direction_pred:
#       self.direction_conv_out = nn.Sequential(
#         nn.Conv2d(448, 256, 3, padding=1, bias=False),
#         nn.BatchNorm2d(256),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(256, 128, 3, padding=1, bias=False),
#         nn.BatchNorm2d(128),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(128, direction_dim+1, 1),
#       )

#   def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll):
#     points_xyz = lidar_data[:, :, :3]
#     points_feature = lidar_data[:, :, 3:]
#     voxels = points_to_voxels(
#       points_xyz, lidar_mask, self.xbound, self.ybound, self.zbound
#     )
#     points_feature = torch.cat(
#       [lidar_data,  # 5
#        torch.unsqueeze(voxels['voxel_point_count'], dim=-1),  # 1
#        voxels['local_points_xyz'],  # 3
#        voxels['point_centroids'],  # 3
#        points_xyz - voxels['voxel_centers'],  # 3
#       ], dim=-1
#     )
#     points_feature = self.pn(points_feature, voxels['points_mask'])
#     voxel_feature = torch_scatter.scatter_mean(
#       points_feature,
#       torch.unsqueeze(voxels['voxel_indices'], dim=1),
#       dim=2,
#       dim_size=voxels['num_voxels'])
#     batch_size = lidar_data.size(0)
#     voxel_feature = voxel_feature.view(batch_size, -1, voxels['grid_size'][0], voxels['grid_size'][1])
#     voxel_feature1 = self.block1(voxel_feature)
#     voxel_feature2 = self.block2(voxel_feature1)
#     voxel_feature3 = self.block3(voxel_feature2)
#     voxel_feature1 = self.up1(voxel_feature1)
#     voxel_feature2 = self.up2(voxel_feature2)
#     voxel_feature3 = self.up3(voxel_feature3)
#     voxel_feature = torch.cat([voxel_feature1, voxel_feature2, voxel_feature3], dim=1)
#     # voxel_feature = self.dropout_lidar(voxel_feature)
#     return self.conv_out(voxel_feature).transpose(3, 2), self.instance_conv_out(voxel_feature).transpose(3, 2), self.direction_conv_out(voxel_feature).transpose(3, 2)


class PointPillarEncoder(nn.Module):
    def __init__(self, C, xbound, ybound, zbound):
        super(PointPillarEncoder, self).__init__()
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.pn = PointNet(15, 64)
        self.block1 = PillarBlock(64, dims=64, num_layers=2, stride=1)
        self.block2 = PillarBlock(64, dims=128, num_layers=3, stride=2)
        self.block3 = PillarBlock(128, 256, num_layers=3, stride=2)
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        convert_relu_to_softplus(self.block1, nn.Hardswish())
        convert_relu_to_softplus(self.block2, nn.Hardswish())
        convert_relu_to_softplus(self.block3, nn.Hardswish())
        convert_relu_to_softplus(self.up1, nn.Hardswish())
        convert_relu_to_softplus(self.up2, nn.Hardswish())
        convert_relu_to_softplus(self.up3, nn.Hardswish())
        self.conv_out = nn.Sequential(
            nn.Conv2d(448, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, C, 1),
        )

    def forward(self, points, points_mask):
        points_xyz = points[:, :, :3]
        points_feature = points[:, :, 3:]
        voxels = points_to_voxels(
          points_xyz, points_mask, self.xbound, self.ybound, self.zbound
        )
        points_feature = torch.cat(
            [points,  # 5
            torch.unsqueeze(voxels['voxel_point_count'], dim=-1),  # 1
            voxels['local_points_xyz'],  # 3
            voxels['point_centroids'],  # 3
            points_xyz - voxels['voxel_centers'],  # 3
            ], dim=-1
        )
        points_feature = self.pn(points_feature, voxels['points_mask'])
        voxel_feature = torch_scatter.scatter_mean(
            points_feature,
            torch.unsqueeze(voxels['voxel_indices'], dim=1),
            dim=2,
            dim_size=voxels['num_voxels'])

        batch_size = points.size(0)
        voxel_feature = voxel_feature.view(batch_size, -1, voxels['grid_size'][0], voxels['grid_size'][1])
        voxel_feature1 = self.block1(voxel_feature)
        voxel_feature2 = self.block2(voxel_feature1)
        voxel_feature3 = self.block3(voxel_feature2)
        voxel_feature1 = self.up1(voxel_feature1)
        voxel_feature2 = self.up2(voxel_feature2)
        voxel_feature3 = self.up3(voxel_feature3)
        voxel_feature_final = torch.cat([voxel_feature1, voxel_feature2, voxel_feature3], dim=1)

        return self.conv_out(voxel_feature_final).transpose(3, 2), voxel_feature.transpose(3, 2)
