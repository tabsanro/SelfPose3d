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

    def forward(self, views=None, meta=None, targets_2d=None, weights_2d=None, targets_3d=None, input_heatmaps=None):
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

        # 2D loss 계산 (training 시에만)
        if self.training:
            criterion = PerJointMSELoss().cuda()
            loss_2d = torch.zeros(1, device=device, requires_grad=True)
            if targets_2d is not None:
                for t, w, o in zip(targets_2d, weights_2d, all_heatmaps):
                    loss_2d = loss_2d + criterion(o, t, True, w)
                loss_2d = loss_2d / len(all_heatmaps)
            result['loss_2d'] = loss_2d
        
        # 2D only training이면 여기서 반환
        if self.train_only_2d:
            return result
        
        # 3D pose estimation
        if self.training:
            criterion = PerJointMSELoss().cuda()
            loss_3d = torch.zeros(1, device=device, requires_grad=True)
        
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
            if self.training and targets_3d is not None:
                loss_3d = criterion(root_cubes, targets_3d)
                result['loss_3d'] = loss_3d

        # Pose prediction
        pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
        pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)

        if self.training:
            criterion_cord = PerJointL1Loss().cuda()
            loss_cord = torch.zeros(1, device=device, requires_grad=True)
            count = 0

        for n in range(self.num_cand):
            index = (pred[:, n, 0, 3] >= 0)
            if torch.sum(index) > 0:
                single_pose = self.pose_net(all_heatmaps, meta, grid_centers[:, n])
                pred[:, n, :, 0:3] = single_pose

                # 3D pose coordinate loss 계산 (training 시에만)
                if self.training and 'joints_3d' in meta[0] and 'joints_3d_vis' in meta[0]:
                    gt_3d = meta[0]['joints_3d'].float()
                    for i in range(batch_size):
                        if pred[i, n, 0, 3] >= 0:
                            person_idx = pred[i, n, 0, 3].long()
                            targets = gt_3d[i:i + 1, person_idx]
                            weights_3d = meta[0]['joints_3d_vis'][i:i + 1, person_idx, :, 0:1].float()
                            count += 1
                            loss_cord = loss_cord + criterion_cord(single_pose[i:i + 1], targets, True, weights_3d)

        if self.training and count > 0:
            loss_cord = loss_cord / count
            result['loss_cord'] = loss_cord

        # 결과 업데이트
        result.update({
            'pred': pred,
            'grid_centers': grid_centers
        })

        return result


def get_multi_person_pose_net(cfg, is_train=True):
    if cfg.BACKBONE_MODEL:
        backbone = eval(cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train)
    else:
        backbone = None
    model = MultiPersonPoseNet(backbone, cfg)
    return model