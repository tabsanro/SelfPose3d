# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from . import pose_resnet
from .cuboid_proposal_net import CuboidProposalNet
from .pose_regression_net import PoseRegressionNet
from ..core.loss import PerJointMSELoss
from ..core.loss import PerJointL1Loss


class MultiPersonPoseNet(nn.Module):
    def __init__(self, backbone, cfg):
        super(MultiPersonPoseNet, self).__init__()
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.num_joints = cfg.NETWORK.NUM_JOINTS

        self.train_only_2d = cfg.NETWORK.TRAIN_ONLY_2D
        self.backbone = backbone
        if not self.train_only_2d:
            self.root_net = CuboidProposalNet(cfg)
            self.pose_net = PoseRegressionNet(cfg)

        self.USE_GT = cfg.NETWORK.USE_GT
        self.root_id = cfg.DATASET.ROOTIDX
        self.dataset_name = cfg.DATASET.TEST_DATASET

    def forward(self, views=None, meta=None, input_heatmaps=None):
        # 입력 처리
        if views is not None:
            device = views[0].device
            batch_size = views[0].shape[0]
            all_heatmaps = []
            for view in views:
                heatmaps = self.backbone(view)
                all_heatmaps.append(heatmaps)
        else:
            all_heatmaps = input_heatmaps
            device = all_heatmaps[0].device
            batch_size = all_heatmaps[0].shape[0]

        result = {'heatmaps': all_heatmaps}
        
        # 2D only training이면 여기서 반환
        if self.train_only_2d:
            return result
        
        # Root proposal
        if self.USE_GT:
            num_person = meta[0]['num_person']
            grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=device)
            grid_centers[:, :, 0:3] = meta[0]['roots_3d'].float()
            grid_centers[:, :, 3] = -1.0
            for i in range(batch_size):
                grid_centers[i, :num_person[i], 3] = torch.tensor(range(num_person[i]), device=device)
                grid_centers[i, :num_person[i], 4] = 1.0
        else:
            root_cubes, grid_centers = self.root_net(all_heatmaps, meta)
            
            # 3D root loss 계산 (training 시에만)
            if self.training:
                result['root_cubes'] = root_cubes
            else:
                del root_cubes
            result['grid_centers'] = grid_centers
        
        pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
        pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)

        valid_mask = grid_centers[:, :, 3] >= 0  # [batch_size, num_cand]
        if torch.any(valid_mask):
        # 유효한 후보자들의 인덱스 추출
            batch_indices, cand_indices = torch.where(valid_mask)
            if len(batch_indices) > 0:
                # 배치 처리를 위한 데이터 준비
                batch_grid_centers = grid_centers[batch_indices, cand_indices]  # [N, 5]
                batch_poses = self.pose_net(all_heatmaps, meta, batch_grid_centers, batch_indices)  # [N, num_joints, 3]

                pred[batch_indices, cand_indices, :, 0:3] = batch_poses

        # 결과 업데이트
        result['pred'] = pred  # [batch_size, num_cand, num_joints, 5]

        return result


def get_multi_person_pose_net(cfg, is_train=True):
    if cfg.BACKBONE_MODEL:
        backbone = eval(cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train)
    else:
        backbone = None
    model = MultiPersonPoseNet(backbone, cfg)
    return model