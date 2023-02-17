import torch
import torch.nn as nn
import torch.nn.functional as F
from .geometry_utils import three_nn, three_weighted_sum

class PointsetFeaturePropagation(nn.Module):
    """
    Propagate features from an abstracted point set back to the original point set,
    analogous to upsampling followed by 1x1 convolutions on an image grid.
    """
    def __init__(self, dim_feats, mlp):
        super(PointsetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        in_channel = dim_feats
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            in_channel = out_channel

    def forward(self, pos1, pos2, feats1, feats2, fast=True):
        """
        Run PointSetFeaturePropagation.
        Args:
            pos1: input point set position data, [B, C, N]
            pos2: abstracted point set position data, [B, C, S]
            feats1: input point set feature data, [B, D, N]
            feats2: abstracted point set feature data, [B, D, S]
        Returns:
            new_feats: upsampled point set feature data, [B, D', N]
        """
        B, _, N = pos1.shape
        if pos2 is None:
            interpolated_feats = feats2.repeat(1, 1, N)
        else:
            S = pos2.shape[2]
            # get 3 nearest neighbors for interpolation
            nn_dists, nn_indices = three_nn(point_pos=pos2, query_pos=pos1, fast=fast)
            # get interpolation weights
            nn_dists_recip = 1.0 / (nn_dists + 1e-8)
            norm = torch.sum(nn_dists_recip, dim=2, keepdim=True)
            nn_weights = nn_dists_recip / norm
            # interpolate features of 3 nearest neighbors
            interpolated_feats = three_weighted_sum(point_feats=feats2, indices=nn_indices, weights=nn_weights, fast=fast)
        if feats1 is not None:
            new_feats = torch.cat([feats1, interpolated_feats], dim=1)
        else:
            new_feats = interpolated_feats
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feats = F.relu(bn(conv(new_feats)))
        return new_feats