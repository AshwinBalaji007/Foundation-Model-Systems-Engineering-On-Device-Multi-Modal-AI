#!/bin/bash

# ==============================================================================
# Professional Training Launcher Script
# ==============================================================================
#
# This script launches the distributed training job. It should be executed
# from the root of the 'ai-model-pipeline' directory.
#
# Usage:
# ./scripts/run_training.sh

# --- Configuration ---
# Set the number of GPUs to use for training.
# For a single-GPU machine (or a CPU-only test), this should be 1.
# On a multi-GPU cloud instance, you can set this to 2, 4, 8, etc.
NUM_GPUS=1

# --- Launch the Training ---
# We execute 'train_distributed.py' as a file. The script itself is
# responsible for handling its Python path, making this call clean and robust.
echo "============================================================"
echo "🚀 Launching distributed training with $NUM_GPUS GPU(s)..."
echo "============================================================"

torchrun --nproc_per_node=$NUM_GPUS src/training/train_distributed.py

echo "============================================================"
echo "✅ Training script has finished."
echo "============================================================"