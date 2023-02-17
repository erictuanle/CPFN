# Importation of packages
import os
import sys
import torch
import numpy as np
from SPFN.losses_implementation import compute_all_losses
from PointNet2.pointnet2_ops.modules.pointset_abstraction import PointsetAbstraction
from PointNet2.pointnet2_ops.modules.pointset_feature_propagation import PointsetFeaturePropagation

class PointNet2(torch.nn.Module):
    def __init__(self, dim_input=3, dim_pos=3, output_sizes=[16], use_glob_features=False, use_loc_features=False, features_extractor=False):
        super(PointNet2, self).__init__()
        self.dim_pos = dim_pos
        self.use_glob_features = use_glob_features
        self.use_loc_features = use_loc_features
        self.features_extractor = features_extractor
        # Encoding stage
        self.sa1 = PointsetAbstraction(num_points=512, dim_pos=dim_pos, dim_feats=dim_input-dim_pos, radius_list=[0.2], num_samples_list=[64], mlp_list=[[64,64,128]], group_all=False)
        self.sa2 = PointsetAbstraction(num_points=128, dim_pos=dim_pos, dim_feats=128, radius_list=[0.4], num_samples_list=[64], mlp_list=[[128,128,256]], group_all=False)
        self.sa3 = PointsetAbstraction(num_points=None, dim_pos=dim_pos, dim_feats=256, radius_list=None, num_samples_list=None, mlp_list=[256, 512, 1024], group_all=True)
        # Decoding stage
        offset = 0
        if self.use_glob_features:
            offset += 1024
        if self.use_loc_features:
            offset += 128
        self.sfp1 = PointsetFeaturePropagation(dim_feats=1024+offset+256, mlp=[256,256])
        self.sfp2 = PointsetFeaturePropagation(dim_feats=256+128, mlp=[256,128])
        self.sfp3 = PointsetFeaturePropagation(dim_feats=128+dim_input-dim_pos, mlp=[128,128,128])
        # FC stage
        self.fc1 = torch.nn.Conv1d(128, 128, 1)
        if not self.features_extractor:
            self.bn1 = torch.nn.BatchNorm1d(128)
            self.fc2 = torch.nn.ModuleList()
            for output_size in output_sizes:
                self.fc2.append(torch.nn.Conv1d(128, output_size, 1))

    def forward(self, x, glob_features=None, loc_features=None, fast=True):
        x = x.transpose(2,1)
        batch_size = x.shape[0]
        input_pos = x[:,:self.dim_pos,:]
        if x.shape[1] > self.dim_pos:
            input_feats = x[:,self.dim_pos:,:]
        else:
            input_feats = None
        # Encoding stage
        l1_pos, l1_feats = self.sa1(input_pos, input_feats, fast=fast)
        l2_pos, l2_feats = self.sa2(l1_pos, l1_feats, fast=fast)
        l3_pos, l3_feats = self.sa3(l2_pos, l2_feats, fast=fast)
        # Adding additional features
        if self.use_glob_features:
            l3_feats = torch.cat((l3_feats, glob_features.unsqueeze(2)), dim=1)
        if self.use_loc_features:
            l3_feats = torch.cat((l3_feats, loc_features.unsqueeze(2)), dim=1)
        # Decoding stage
        l4_feats = self.sfp1(l2_pos, l3_pos, l2_feats, l3_feats, fast=fast)
        l5_feats = self.sfp2(l1_pos, l2_pos, l1_feats, l4_feats, fast=fast)
        l6_feats = self.sfp3(input_pos, l1_pos, input_feats, l5_feats, fast=fast)
        # FC stage
        output_feat = self.fc1(l6_feats)
        if not self.features_extractor:
            output_feat = torch.nn.functional.relu(self.bn1(output_feat))
            output_feat = torch.nn.functional.dropout(output_feat, p=0.5)
            results = []
            for fc2_layer in self.fc2:
                result = fc2_layer(output_feat)
                result = result.transpose(1,2)
                results.append(result)
            results.append(l3_feats)
            results.append(output_feat)
            return results
        else:
            return l3_feats, output_feat