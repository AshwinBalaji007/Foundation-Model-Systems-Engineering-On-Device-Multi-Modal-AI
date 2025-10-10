# ==============================================================================
# AI Systems Engineering: Distributed Training Script (for BLIP)
# ==============================================================================
import sys
from pathlib import Path
src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

import os
import yaml
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
# --- THE CHANGE ---
from transformers import AutoProcessor, AutoModelForConditionalGeneration # BLIP is a conditional generation model
import functools
from tqdm import tqdm
from data.dataset import get_dataloader

def setup(rank: int, world_size: int):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank: int, world_size: int, config: dict):
    if rank == 0:
        print("--- Initializing Training on All Ranks ---")
    setup(rank, world_size)

    model_name = config['model']['base_model_name']
    if rank == 0: print(f"Loading base model '{model_name}'...")
    processor = AutoProcessor.from_pretrained(model_name)
    
    # --- THE CHANGE ---
    # BLIP uses a different AutoModel class and doesn't need CPU Offload on a T4 GPU.
    model = AutoModelForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 # Standard float16 is fine for BLIP
    )
    
    model.to(rank) # Move the smaller model directly to the GPU
    model = FSDP(model) # A simpler FSDP wrapper is sufficient
    
    if rank == 0: print("✅ Model successfully loaded and wrapped with FSDP.")

    if rank == 0: print("Creating distributed dataloaders...")
    dataloader = get_dataloader(
        rank=rank, world_size=world_size,
        dataset_name=config['data']['dataset_name'],
        processor=processor,
        batch_size=config['training']['batch_size_per_device']
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    if rank == 0: print("\n--- Starting Training Loop ---")
    for epoch in range(config['training']['num_epochs']):
        if rank == 0: print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        dataloader.sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, disable=(rank != 0), desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            batch = {k: v.to(rank) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if rank == 0: progress_bar.set_postfix(loss=loss.item())
        dist.barrier()

    if rank == 0:
        print("\n--- Training Complete ---")
        save_path = config['model']['fine_tuned_path']
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"✅ Model checkpoint would be saved to: {save_path}")

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