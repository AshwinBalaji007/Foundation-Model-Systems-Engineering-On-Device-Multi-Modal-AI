# ==============================================================================
# AI Systems Engineering: Distributed Training Script (VRAM Optimized)
# ==============================================================================
#
# This script fine-tunes a multi-modal foundation model using PyTorch's
# Fully Sharded Data Parallel (FSDP) with advanced memory-saving strategies
# to enable training large models on memory-constrained GPUs.

import sys
from pathlib import Path

# --- FOOLPROOF PYTHON PATH FIX ---
src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))
# --- END OF FIX ---

import os
import yaml
import torch
import torch.distributed as dist
# --- NEW, ADVANCED FSDP IMPORTS ---
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
# --- END OF NEW IMPORTS ---
from transformers import AutoProcessor, AutoModelForVision2Seq
import functools
from tqdm import tqdm

# Now that the path is fixed, this import will work perfectly.
from data.dataset import get_dataloader

# --- Distributed Training Setup Functions ---

def setup(rank: int, world_size: int):
    """Initializes the distributed process group using environment variables."""
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

    # --- 1. Load Model and Processor (on CPU first) ---
    model_name = config['model']['base_model_name']
    
    if rank == 0:
        print(f"Loading base model '{model_name}' onto CPU...")

    processor = AutoProcessor.from_pretrained(model_name)
    
    # --- THE VRAM (GPU MEMORY) FIX IS HERE ---
    # We load the model onto the CPU first. FSDP will manage moving the
    # sharded pieces to the GPU as needed. We remove `device_map="auto"`.
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    )

    # --- 2. Define a Memory-Saving FSDP Strategy ---
    # FULL_SHARD: Shards not just the model parameters, but also gradients
    # and optimizer states. This is the most memory-efficient strategy.
    fsdp_sharding_strategy = ShardingStrategy.FULL_SHARD
    
    # CPU Offloading: The ultimate memory saver. It moves sharded parameters
    # to the CPU's RAM when they are not being used in the forward/backward pass.
    # This slows down training but is often necessary for large models on smaller GPUs.
    cpu_offload = CPUOffload(offload_params=True)

    # --- 3. Wrap the Model with the FSDP Strategy ---
    # Define how to wrap the layers of the model.
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e8)
    
    # Apply FSDP to the model with our memory-saving strategy.
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=fsdp_sharding_strategy,
        cpu_offload=cpu_offload, # Enable CPU offloading
        device_id=torch.cuda.current_device() # Important to specify the device ID
    )
    
    if rank == 0:
        print("✅ Model successfully wrapped with FSDP using FULL_SHARD and CPU Offload strategy.")

    # --- 4. Create the Dataloader ---
    if rank == 0:
        print("Creating distributed dataloaders...")
    dataloader = get_dataloader(
        rank=rank,
        world_size=world_size,
        dataset_name=config['data']['dataset_name'],
        processor=processor,
        batch_size=config['training']['batch_size_per_device']
    )

    # --- 5. Setup Optimizer ---
    # The optimizer must be created AFTER the model is wrapped in FSDP.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # --- 6. The Training Loop ---
    if rank == 0:
        print("\n--- Starting Training Loop ---")

    for epoch in range(config['training']['num_epochs']):
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        dataloader.sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, disable=(rank != 0), desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            # Move batch to the correct GPU
            batch = {k: v.to(rank) for k, v in batch.items()}
            
            # Forward, backward, and optimizer steps are the same
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if rank == 0:
                progress_bar.set_postfix(loss=loss.item())

        dist.barrier()

    # --- 7. Save the Final Model ---
    if rank == 0:
        print("\n--- Training Complete ---")
        # Saving an FSDP model requires gathering the shards.
        # This is a complex step that we simulate for now.
        save_path = config['model']['fine_tuned_path']
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"✅ Model checkpoint would be saved to: {save_path}")

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