# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import argparse
import os
import time
from tqdm import tqdm
from prettytable import PrettyTable
import copy
import logging

# import _init_paths
from SelfPose3d.core.config import config
from SelfPose3d.core.config import update_config
from SelfPose3d.utils.utils import create_logger, load_backbone_panoptic
from SelfPose3d import dataset
from SelfPose3d import models
from SelfPose3d.utils.vis import save_batch_heatmaps_multi, save_debug_3d_images_all


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        '--test-file', help='test_file', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def main():
    args = parse_args()
    cfg_name = os.path.basename(args.cfg).split('.')[0]
    final_output_dir = os.path.join("./results_publ/", cfg_name)
    os.makedirs(final_output_dir, exist_ok=True)                

    gpus = [int(i) for i in config.GPUS.split(',')]
    #gpus = [0]
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config, is_train=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    if os.path.isfile(args.test_file):
        test_model_file = args.test_file
    else:
        test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')

    model.eval()
    results_all = []
    with torch.no_grad():
        for i, (
            inputs,
            targets_2d,
            weights_2d,
            targets_3d,
            meta,
            input_heatmap,
        ) in enumerate(test_loader):
            pred, _, grid_centers, _, _, _ = model(views=inputs, meta=meta)
            pred = pred.cpu().numpy()
            grid_centers = grid_centers.cpu().numpy()
            result = {
                'pred': pred,
                'root': grid_centers,
            }
            results_all.append(result)
            if i % 100 == 0:
                print('processing {} / {}'.format(i, len(test_loader)))
    with open('results.pkl', 'wb') as f:
        import pickle
        pickle.dump(results_all, f)
    print('results saved to {}'.format('results.pkl'))


if __name__ == "__main__":
    main()
