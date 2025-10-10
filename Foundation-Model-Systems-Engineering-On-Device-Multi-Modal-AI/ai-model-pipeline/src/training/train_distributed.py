# ==============================================================================
# AI Systems Engineering: Distributed Training Script
# ==============================================================================
#
# This script fine-tunes a multi-modal foundation model using PyTorch's
# Fully Sharded Data Parallel (FSDP) for efficient multi-GPU training.
# It is designed to be launched via `torchrun`.

import sys
from pathlib import Path

# --- FOOLPROOF PYTHON PATH FIX ---
# This robustly adds the project's 'src' directory to the Python path.
# It allows the script to be run from anywhere and still find its modules.
src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))
# --- END OF FIX ---

import os
import yaml
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoProcessor, AutoModelForVision2Seq
import functools
from tqdm import tqdm

# Now that the path is fixed, this import will work perfectly.
from data.dataset import get_dataloader

# --- Distributed Training Setup Functions ---

def setup(rank: int, world_size: int):
    """Initializes the distributed process group using environment variables."""
    # torchrun automatically sets MASTER_ADDR and MASTER_PORT
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

    # --- 1. Load Model and Processor ---
    model_name = config['model']['base_model_name']
    
    if rank == 0:
        print(f"Loading base model '{model_name}'...")

    processor = AutoProcessor.from_pretrained(model_name)
    
    # --- THE OOM (OUT-OF-MEMORY) FIX IS HERE ---
    # Use device_map="auto" to intelligently load the large model directly
    # onto the available GPU(s), bypassing the need to fit it all into CPU RAM first.
    # This is critical for loading models larger than your system's RAM.
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # bfloat16 is optimized for modern GPUs
        device_map="auto" 
    )

    # Note: Because we use device_map, the model is already on the GPU.

    # --- 2. Wrap the Model with FSDP ---
    # FSDP will now take the model that's already on the GPU(s) and shard it for training.
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e8)
    model = FSDP(model, auto_wrap_policy=auto_wrap_policy, device_id=torch.cuda.current_device())
    
    if rank == 0:
        print("✅ Model successfully loaded and wrapped with FSDP.")

    # --- 3. Create the Dataloader ---
    if rank == 0:
        print("Creating distributed dataloaders...")
    dataloader = get_dataloader(
        rank=rank,
        world_size=world_size,
        dataset_name=config['data']['dataset_name'],
        processor=processor,
        batch_size=config['training']['batch_size_per_device']
    )

    # --- 4. Setup Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # --- 5. The Training Loop ---
    if rank == 0:
        print("\n--- Starting Training Loop ---")

    for epoch in range(config['training']['num_epochs']):
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Set the epoch for the sampler to ensure data is shuffled differently each epoch
        dataloader.sampler.set_epoch(epoch)
        
        progress_bar = tqdm(dataloader, disable=(rank != 0), desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            # Move batch to the correct GPU for this process
            batch = {k: v.to(rank) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass (FSDP handles gradient synchronization)
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress bar description on the main process
            if rank == 0:
                progress_bar.set_postfix(loss=loss.item())

        # Barrier to ensure all processes finish the epoch before proceeding
        dist.barrier()

    # --- 6. Save the Final Model ---
    if rank == 0:
        print("\n--- Training Complete ---")
        print("Saving final model checkpoint...")
        # In a real scenario, you would implement the proper FSDP state dict saving logic here.
        save_path = config['model']['fine_tuned_path']
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"✅ Model checkpoint would be saved to: {save_path}")

    cleanup()

# --- Script Entry Point ---
if __name__ == '__main__':
    # --- 1. Load Configuration ---
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "configs" / "training_config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # --- 2. Get Distributed Training Environment Variables ---
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # --- 3. Validate Environment and Start Training ---
    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires at least one CUDA-enabled GPU.")
        
    if world_size > torch.cuda.device_count():
         raise RuntimeError(f"Requested {world_size} GPUs, but only {torch.cuda.device_count()} are available.")
    
    # Call the main training function for the current process
    train(rank, world_size, config)