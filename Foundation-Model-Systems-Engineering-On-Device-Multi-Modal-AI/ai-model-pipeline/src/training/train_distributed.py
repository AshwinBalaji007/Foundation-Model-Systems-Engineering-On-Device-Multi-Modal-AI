# ==============================================================================
# AI Systems Engineering: Distributed Training Script (Final, Stable Version)
# ==============================================================================
# This version includes a Gradient Scaler for stable mixed-precision (float16) training.

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
# --- NEW IMPORTS FOR STABLE TRAINING ---
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoProcessor, BlipForConditionalGeneration
from data.dataset import get_dataloader

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def train(rank: int, world_size: int, config: dict):
    if rank == 0:
        print("--- Initializing Training on All Ranks ---")
    
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    model_name = config['model']['base_model_name']
    if rank == 0: print(f"Loading base model '{model_name}'...")
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    
    model.to(device)
    model = DDP(model, device_ids=[local_rank])
    
    if rank == 0: print("✅ Model successfully loaded and wrapped with DDP.")

    if rank == 0: print("Creating distributed dataloaders...")
    dataloader = get_dataloader(
        rank=rank, world_size=world_size,
        dataset_name=config['data']['dataset_name'], processor=processor,
        batch_size=config['training']['batch_size_per_device']
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    
    # --- THE GRADIENT SCALER FIX ---
    # Initialize a gradient scaler for mixed-precision training
    scaler = GradScaler()
    
    if rank == 0: print("\n--- Starting Training Loop with Gradient Scaling ---")
    model.train()
    for epoch in range(config['training']['num_epochs']):
        dataloader.sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, disable=(rank != 0), desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            batch["labels"] = batch["input_ids"]
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # --- Use autocast for the forward pass ---
            # This automatically runs operations in float16 where possible
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
            
            # --- Scale the loss and perform the backward pass ---
            scaler.scale(loss).backward()
            
            # --- Unscale gradients and perform optimizer step ---
            scaler.step(optimizer)
            
            # --- Update the scaler for the next iteration ---
            scaler.update()
            
            if rank == 0: progress_bar.set_postfix(loss=loss.item())
        dist.barrier()

    if rank == 0:
        print("\n--- Training Complete ---")
        save_path = config['model']['fine_tuned_path']
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.module.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"✅ Model and processor saved to: {save_path}")

    cleanup()

# --- (The '__main__' block remains the same) ---
if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "configs" / "training_config.yaml"
    with open(config_path, 'r') as file: config = yaml.safe_load(file)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires at least one CUDA-enabled GPU.")
    if world_size > torch.cuda.device_count():
         raise RuntimeError(f"Requested {world_size} GPUs, but only {torch.cuda.device_count()} are available.")
    train(rank, world_size, config)