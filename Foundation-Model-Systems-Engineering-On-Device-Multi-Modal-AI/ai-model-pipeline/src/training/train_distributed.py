# ==============================================================================
# AI Systems Engineering: Distributed Training Script (for Image Classification)
# ==============================================================================
#
# This script trains a standard computer vision model (ResNet) on the CIFAR10
# dataset using PyTorch's DistributedDataParallel (DDP).

import sys
from pathlib import Path

# --- FOOLPROOF PYTHON PATH FIX ---
src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))
# --- END OF FIX ---

import os
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models # We'll use a pre-built model from torchvision
from tqdm import tqdm

# Now that the path is fixed, this import will work perfectly.
from data.dataset import get_dataloader

# --- Distributed Training Setup Functions ---

def setup(rank: int, world_size: int):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

# --- Main Training Function ---

def train(rank: int, world_size: int, config: dict):
    """The main training function executed on each GPU process."""
    if rank == 0:
        print("--- Initializing Training on All Ranks ---")
    
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # --- 1. Load the Model ---
    # We are now using a ResNet18 model, perfect for CIFAR10.
    # num_classes=10 because CIFAR10 has 10 categories.
    model = models.resnet18(num_classes=10)
    model.to(device)
    
    # --- 2. Wrap the Model with DDP ---
    # For a model of this size, DDP is the standard and correct choice.
    # It is simpler and more efficient than FSDP for smaller models.
    model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        print("✅ Model successfully wrapped with DDP.")

    # --- 3. Create the Dataloader ---
    if rank == 0:
        print("Creating distributed dataloaders for CIFAR10...")
    dataloader = get_dataloader(
        rank=rank,
        world_size=world_size,
        batch_size=config['training']['batch_size_per_device']
    )

    # --- 4. Setup Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['learning_rate'], momentum=0.9)
    
    # --- 5. The Training Loop ---
    if rank == 0:
        print("\n--- Starting Training Loop ---")

    model.train()
    for epoch in range(config['training']['num_epochs']):
        dataloader.sampler.set_epoch(epoch) # Important for shuffling in distributed training
        progress_bar = tqdm(dataloader, disable=(rank != 0), desc=f"Epoch {epoch+1}")
        
        for images, labels in progress_bar:
            # Move data to the correct GPU
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if rank == 0:
                progress_bar.set_postfix(loss=loss.item())

        dist.barrier()

    if rank == 0:
        print("\n--- Training Complete ---")
        save_path = config['model']['fine_tuned_path']
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # In DDP, we save the 'module' state_dict to get the raw model weights
        torch.save(model.module.state_dict(), save_path)
        print(f"✅ Model checkpoint saved to: {save_path}")

    cleanup()

# --- (The '__main__' block remains the same) ---
if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "configs" / "training_config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires at least one CUDA-enabled GPU.")
        
    if world_size > torch.cuda.device_count():
         raise RuntimeError(f"Requested {world_size} GPUs, but only {torch.cuda.device_count()} are available.")
    
    train(rank, world_size, config)