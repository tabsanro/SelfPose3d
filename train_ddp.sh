#!/bin/bash

# DDP training script for SelfPose3d
# Usage: bash train_ddp.sh [config_file] [num_gpus]

CONFIG_FILE=${1:-"configs/human36m/h36m.yaml"}
NUM_GPUS=${2:-4}

echo "Starting DDP training with $NUM_GPUS GPUs"
echo "Config file: $CONFIG_FILE"

# Method 1: Using torchrun (recommended for PyTorch 1.10+)
torchrun --nproc_per_node=$NUM_GPUS \
         --master_port=29500 \
         tools/train_3d_ddp.py \
         --cfg $CONFIG_FILE

# Method 2: Alternative using python -m torch.distributed.launch (for older PyTorch versions)
# python -m torch.distributed.launch \
#        --nproc_per_node=$NUM_GPUS \
#        --master_port=29500 \
#        tools/train_3d_ddp.py \
#        --cfg $CONFIG_FILE
