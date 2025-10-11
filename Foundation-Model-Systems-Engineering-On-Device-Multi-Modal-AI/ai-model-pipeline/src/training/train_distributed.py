# ==============================================================================
# AI Systems Engineering: Distributed Training Script (Final Bare Metal Version)
# ==============================================================================
# This version removes manual mixed-precision handling to work with models
# that internally cast to float16.

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
    
    # --- THE DEFINITIVE FIX IS HERE ---
    # 1. Load the model WITH torch_dtype=torch.float16.
    # We will now fully commit to the float16 workflow that the model prefers.
    model = BlipForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    
    # 2. Move the float16 model to the GPU.
    model.to(device)
    
    # 3. Wrap the model with DDP.
    model = DDP(model, device_ids=[local_rank])
    
    if rank == 0: print("✅ Model successfully loaded in FP16 and wrapped with DDP.")

    if rank == 0: print("Creating distributed dataloaders...")
    dataloader = get_dataloader(
        rank=rank,
        world_size=world_size,
        dataset_name=config['data']['dataset_name'], processor=processor,
        batch_size=config['training']['batch_size_per_device']
    )

    # 4. Use a standard optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    
    # 5. REMOVE the GradScaler and autocast. We are now operating fully in float16.
    
    if rank == 0: print("\n--- Starting Training Loop (Native FP16) ---")
    model.train()
    for epoch in range(config['training']['num_epochs']):
        dataloader.sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, disable=(rank != 0), desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            batch["labels"] = batch["input_ids"]
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # The forward pass now runs natively in float16.
            outputs = model(**batch)
            loss = outputs.loss
            
            # Standard backward pass and optimizer step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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