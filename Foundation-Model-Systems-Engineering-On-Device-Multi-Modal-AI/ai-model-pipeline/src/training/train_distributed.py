# ==============================================================================
# AI Systems Engineering: Distributed Training Script (Final Stable Version)
# ==============================================================================
# This version implements the standard, industry-best-practice pattern for
# stable mixed-precision training using a GradScaler and autocast.

import sys
from pathlib import Path
src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

import os
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# --- Import the necessary tools for stable mixed precision ---
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoProcessor, BlipForConditionalGeneration
from data.dataset import get_dataloader

def setup():
    """Initializes the distributed process group."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def train(rank: int, world_size: int, config: dict):
    if rank == 0:
        print("--- Initializing Training on All Ranks ---")
    
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    model_name = config['model']['base_model_name']
    if rank == 0: print(f"Loading base model '{model_name}' in full FP32 precision...")
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    # --- THE DEFINITIVE FIX IS HERE ---
    # 1. Load the model in standard float32 precision. Do NOT use torch_dtype.
    # This keeps a stable "master copy" of the weights.
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    
    # 2. Move the full-precision model to the GPU.
    model.to(device)
    
    # 3. Wrap the model with DDP.
    model = DDP(model, device_ids=[local_rank])
    
    if rank == 0: print(" Model successfully loaded and wrapped with DDP.")

    if rank == 0: print("Creating distributed dataloaders...")
    dataloader = get_dataloader(
        rank=rank,
        world_size=world_size,
        dataset_name=config['data']['dataset_name'], processor=processor,
        batch_size=config['training']['batch_size_per_device']
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    
    # 4. Initialize the GradScaler.
    scaler = GradScaler()
    
    if rank == 0: print("\n--- Starting Training Loop with Stable Mixed Precision ---")
    model.train()
    for epoch in range(config['training']['num_epochs']):
        dataloader.sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, disable=(rank != 0), desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            batch["labels"] = batch["input_ids"]
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # 5. Use autocast ONLY for the forward pass and loss calculation.
            # This performs these operations in float16 for speed/memory.
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
            
            # 6. The scaler handles the rest: scaling the loss, backward pass,
            #    unscaling gradients, and updating the optimizer.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if rank == 0: progress_bar.set_postfix(loss=loss.item())
        dist.barrier()

    if rank == 0:
        print("\n--- Training Complete ---")
        save_path = config['model']['fine_tuned_path']
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.module.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f" Model and processor saved to: {save_path}")

    cleanup()

# --- (The __main__ block remains the same) ---
if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "configs" / "training_config.yaml"
    with open(config_path, 'r') as file: config = yaml.safe_load(file)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires at least one CUDA-enabled GPU.")
    if world_size > torch.cuda.device_count():
         raise RuntimeError(f"Requested {world_size} GPUs, but only {torch.cuda.device_count()} are available.")
    train(rank, world_size, config)