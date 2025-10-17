# ==============================================================================
# AI Systems Engineering: Distributed Training Script (Definitive Final Version)
# ==============================================================================
# This script fine-tunes a VisionEncoderDecoderModel using a robust data pipeline
# and a stable, mixed-precision training loop with PyTorch DDP.

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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# --- Import the correct, modern tools ---
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
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
    
    # --- Load the model and its components ---
    # We load the tokenizer and image processor separately, just like in dataset.py
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    image_processor = ViTImageProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Critical for training: set the special tokens for the decoder
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Move the full-precision model to the GPU
    model.to(device)
    
    # Wrap the model with DDP
    model = DDP(model, device_ids=[local_rank])
    
    if rank == 0: print("✅ Model successfully loaded and wrapped with DDP.")

    if rank == 0: print("Creating distributed dataloaders...")
    # Pass the separate components to the dataloader
    dataloader = get_dataloader(
        rank=rank,
        world_size=world_size,
        dataset_name=config['data']['dataset_name'],
        tokenizer=tokenizer,
        image_processor=image_processor,
        batch_size=config['training']['batch_size_per_device']
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    scaler = GradScaler()
    
    if rank == 0: print("\n--- Starting Training Loop with Stable Mixed Precision ---")
    model.train()
    for epoch in range(config['training']['num_epochs']):
        dataloader.sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, disable=(rank != 0), desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            # The model expects "labels" to calculate the loss. For this task,
            # the labels are the same as the input_ids.
            batch["labels"] = batch["input_ids"]
            
            # Move the entire batch to the correct GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Use autocast for the forward pass for speed and memory efficiency
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
            
            # Scaler handles the backward pass and optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if rank == 0: progress_bar.set_postfix(loss=loss.item())
        dist.barrier()

    if rank == 0:
        print("\n--- Training Complete ---")
        save_path = config['model']['fine_tuned_path']
        os.makedirs(save_path, exist_ok=True)
        
        # It's best practice to save the model, tokenizer, and image processor together
        model.module.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        image_processor.save_pretrained(save_path)
        
        print(f"✅ Model and processor components saved to: {save_path}")

    cleanup()

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