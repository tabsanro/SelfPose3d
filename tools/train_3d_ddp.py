'''
Project: SelfPose3d - DDP Version
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import argparse
import os
import pprint
import logging
import json

# import _init_paths
from SelfPose3d.core.config import config
from SelfPose3d.core.config import update_config
from SelfPose3d.core.function import train_3d, train_3d_ssv, validate_3d
from SelfPose3d.utils.utils import create_logger
from SelfPose3d.utils.utils import save_checkpoint, load_checkpoint, load_model_state
from SelfPose3d.utils.utils import load_backbone_panoptic
from SelfPose3d import dataset
from SelfPose3d import models
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network with DDP")
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank for distributed training")
    parser.add_argument("--world-size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def get_optimizer(model):
    lr = config.TRAIN.LR
    with_root_net = not config.NETWORK.USE_GT # USE_GT means using GT proposals for pose regression
    freeze_root_net = config.NETWORK.FREEZE_ROOTNET
    train_backbone = config.NETWORK.TRAIN_BACKBONE
    
    if train_backbone:
        if model.module.backbone is not None:
            for params in model.module.backbone.parameters():
                params.requires_grad = True
    else:
        if model.module.backbone is not None:
            for params in model.module.backbone.parameters():
                params.requires_grad = False
                
    if not config.NETWORK.TRAIN_ONLY_2D:
        if not config.NETWORK.TRAIN_ONLY_ROOTNET:
            for params in model.module.pose_net.parameters():
                params.requires_grad = True
        if with_root_net:
            if freeze_root_net:
                for params in model.module.root_net.parameters():
                    params.requires_grad = False
            else:
                for params in model.module.root_net.parameters():
                    params.requires_grad = True
                    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)
    
    return model, optimizer


def main():
    args = parse_args()
    
    # Initialize distributed mode
    init_distributed_mode(args)
    
    if args.distributed:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cuda')
    
    # Set seed
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    with_ssv = config.WITH_SSV # whether using self-supervised learning
    
    # Only create logger on master process
    if not args.distributed or args.rank == 0:
        logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, "train")
        logger.info(pprint.pformat(args))
        logger.info(pprint.pformat(config))
        
        writer_dict = {
            "writer": SummaryWriter(log_dir=tb_log_dir),
            "train_global_steps": 0,
            "valid_global_steps": 0,
        }
    else:
        logger = None
        final_output_dir = None
        writer_dict = None

    print("=> Loading data ..")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_dataset = eval("dataset." + config.DATASET.TRAIN_DATASET)(
        config,
        config.DATASET.TRAIN_SUBSET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
    )

    # Use DistributedSampler for training
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, 
            num_replicas=args.world_size, 
            rank=args.rank
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,  # Per-GPU batch size
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = eval("dataset." + config.DATASET.TEST_DATASET)(
        config,
        config.DATASET.TEST_SUBSET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
    )

    # Use DistributedSampler for testing (but don't shuffle)
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False
        )
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE,  # Per-GPU batch size
        shuffle=False,
        sampler=test_sampler,
        num_workers=config.WORKERS,
        pin_memory=True,
    )

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print("=> Constructing models ..")
    model = eval("models." + config.MODEL + ".get_multi_person_pose_net")(config, is_train=True)
    
    # Move model to GPU
    model = model.to(device)
    
    # Wrap model with DDP
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=True  # Set to False if all parameters are used
        )
    else:
        # Fallback to single GPU
        model = torch.nn.DataParallel(model).cuda()
    
    print("=> model ->", model)

    model, optimizer = get_optimizer(model)

    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH
    last_epoch = -1
    best_precision = 0

    # Load pretrained backbone
    if config.NETWORK.PRETRAINED_BACKBONE:
        if config.NETWORK.PRETRAINED_BACKBONE_PSEUDOGT:
            print("=> loading backbone from =", config.NETWORK.PRETRAINED_BACKBONE)
            st_dict = {
                k.replace("backbone.", ""): v
                for k, v in torch.load(config.NETWORK.PRETRAINED_BACKBONE, map_location=device).items()
                if "backbone" in k
            }
            mk, uk = model.module.backbone.load_state_dict(st_dict, strict=True)
            print("=> missing keys in backbone =", mk)
            print("=> unexpected keys in backbone =", uk)
        else:
            model = load_backbone_panoptic(model, config.NETWORK.PRETRAINED_BACKBONE)

    # Load pretrained rootnet
    if config.NETWORK.INIT_ROOTNET:
        print("=> loading rootnet from =", config.NETWORK.INIT_ROOTNET)
        st_dict = {
            k.replace("root_net.", ""): v
            for k, v in torch.load(config.NETWORK.INIT_ROOTNET, map_location=device).items()
            if "root_net" in k
        }
        mk, uk = model.module.root_net.load_state_dict(st_dict, strict=True)
        print("=> missing keys in rootnet =", mk)
        print("=> unexpected keys in rootnet=", uk)

    # Load all pretrained weights
    if config.NETWORK.INIT_ALL:
        print("=> loading all from =", config.NETWORK.INIT_ALL)
        st_dict = torch.load(config.NETWORK.INIT_ALL, map_location=device)
        mk, uk = model.module.load_state_dict(st_dict, strict=True)
        print("=> missing keys in all =", mk)
        print("=> unexpected keys in all =", uk)

    # Resume training
    if config.TRAIN.RESUME and (not args.distributed or args.rank == 0):
        start_epoch, model, optimizer, best_precision, last_epoch = load_checkpoint(
            model, optimizer, final_output_dir
        )

    print("=> Training...")
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch=last_epoch
    )

    for epoch in range(start_epoch, end_epoch):
        # Set epoch for distributed sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)
            
        print("Epoch: {}".format(epoch))
        if logger:
            logger.info("learning rate for this epoch {}".format(lr_scheduler.get_last_lr()))

        # Training
        if with_ssv:
            train_3d_ssv(
                config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict
            )
        else:
            train_3d(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict)
            
        lr_scheduler.step()

        # Validation
        if config.NETWORK.TRAIN_ONLY_2D:
            precision = None
        else:
            precision = validate_3d(
                config, model, test_loader, epoch, final_output_dir, with_ssv=with_ssv
            )

        # Save checkpoint (only on master process)
        if not args.distributed or args.rank == 0:
            if precision is not None and precision > best_precision:
                best_precision = precision
                best_model = True
            else:
                best_model = False

            if logger:
                logger.info("=> saving checkpoint to {} (Best: {})".format(final_output_dir, best_model))
                
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.module.state_dict(),
                    "precision": best_precision,
                    "optimizer": optimizer.state_dict(),
                },
                best_model,
                final_output_dir,
            )

    # Save final model (only on master process)
    if not args.distributed or args.rank == 0:
        final_model_state_file = os.path.join(final_output_dir, "final_state.pth.tar")
        if logger:
            logger.info("saving final model state to {}".format(final_model_state_file))
        torch.save(model.module.state_dict(), final_model_state_file)

        if writer_dict:
            writer_dict["writer"].close()

    # Clean up distributed training
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
