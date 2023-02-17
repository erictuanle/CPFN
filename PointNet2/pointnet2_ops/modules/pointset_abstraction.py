from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from .geometry_utils import farthest_point_sample, select_point_subset, ball_query

class PointsetAbstraction(nn.Module):
    """
    Abstract a point set (possibly with features) into a smaller point set,
    analogous to a strided convolution on an image grid.
    """
    def __init__(self, num_points, dim_pos, dim_feats, radius_list, num_samples_list, mlp_list, group_all=False):
        super(PointsetAbstraction, self).__init__()
        self.num_points = num_points
        self.group_all = group_all
        self.radius_list = radius_list if isinstance(radius_list, Sequence) else [radius_list]
        self.num_samples_list = num_samples_list if isinstance(num_samples_list, Sequence) else [num_samples_list]
        self.mlp_list = mlp_list if isinstance(mlp_list[0], Sequence) else [mlp_list]
        if len(self.radius_list) != len(self.num_samples_list) or len(self.radius_list) != len(self.mlp_list):
            raise ValueError('Radius, number of samples and mlps lists must have the same number of entries.')
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(self.mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            in_channel = dim_pos + dim_feats
            for out_channel in self.mlp_list[i]:
                convs.append(nn.Conv2d(in_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                in_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, pos, feats, fast=True):
        """
        Args:
            pos: input point set position data, [B, C, N]
            feats: input point set feature data, [B, D, N]
        Returns:
            new_pos: abstracted point set position data, [B, C, S]
            new_feats: abstracted point set feature data, [B, D', S]
        """
        B, C, N = pos.shape
        S = self.num_points
        if self.group_all:
            subsampling_indices = None
            new_pos = None
        else:
            subsampling_indices = farthest_point_sample(pos, S, fast=fast)
            new_pos = select_point_subset(pos, subsampling_indices)
        new_feats_list = []
        for i, r in enumerate(self.radius_list):
            if self.group_all:
                grouped_pos = pos.view(B, C, 1, N)
                if feats is not None:
                    grouped_feats = torch.cat([grouped_pos, feats.view(B, -1, 1, N)], dim=1)
                else:
                    grouped_feats = grouped_pos
            else:
                K = self.num_samples_list[i]
                group_idx = ball_query(r, K, pos, new_pos, fast=fast)
                grouped_pos = select_point_subset(pos, group_idx)
                grouped_pos -= new_pos.view(B, C, S, 1)
                if feats is not None:
                    grouped_feats = select_point_subset(feats, group_idx)
                    grouped_feats = torch.cat([grouped_feats, grouped_pos], dim=1)
                else:
                    grouped_feats = grouped_pos
            # grouped_feats = grouped_feats.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_feats = F.relu(bn(conv(grouped_feats.contiguous()))) # grouped_feats: [B, D, S, K]
            new_feats = torch.max(grouped_feats, dim=3)[0]  # new_feats: [B, D', S]
            new_feats_list.append(new_feats)
        new_feats = torch.cat(new_feats_list, dim=1)
        return new_pos, new_feats