#!/bin/bash
NUM_GPUS=1 
echo "Launching distributed training with $NUM_GPUS GPUs..."
torchrun --nproc_per_node=$NUM_GPUS src/training/train_distributed.py
echo "Training script has finished."