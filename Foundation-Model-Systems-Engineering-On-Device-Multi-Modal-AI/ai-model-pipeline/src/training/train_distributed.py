# ==============================================================================
# AI Systems Engineering: Distributed Training Script (Final Version for BLIP)
# ==============================================================================
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

# --- THE DEFINITIVE FIX IS HERE ---
# We import the specific, correct model class for the BLIP architecture.
from transformers import AutoProcessor, BlipForConditionalGeneration
# --- END OF FIX ---

from data.dataset import get_dataloader

def setup():
    """Initializes the distributed process group via environment variables."""
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
    if rank == 0: print(f"Loading base model '{model_name}'...")
    
    # Processor loading remains the same
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Use the specific, correct model class
    model = BlipForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    
    model.to(device)
    # For a model of this size, DDP is simpler and more efficient than FSDP.
    model = DDP(model, device_ids=[local_rank])
    
    if rank == 0: print("✅ Model successfully loaded and wrapped with DDP.")

    if rank == 0: print("Creating distributed dataloaders...")
    dataloader = get_dataloader(
        rank=rank,
        world_size=world_size,
        dataset_name=config['data']['dataset_name'],
        processor=processor,
        batch_size=config['training']['batch_size_per_device']
    )

    # For image captioning, the loss is calculated internally by the model
    # when 'labels' are provided, so we don't need a separate criterion.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    if rank == 0: print("\n--- Starting Training Loop ---")
    model.train()
    for epoch in range(config['training']['num_epochs']):
        dataloader.sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, disable=(rank != 0), desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            # The BLIP model expects 'labels' which are the same as 'input_ids' for captioning
            batch["labels"] = batch["input_ids"]
            
            # Move data to the correct GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if rank == 0: progress_bar.set_postfix(loss=loss.item())
        dist.barrier()

    if rank == 0:
        print("\n--- Training Complete ---")
        save_path = config['model']['fine_tuned_path']
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # It's best practice to save the processor along with the model
        model.module.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"✅ Model and processor saved to: {save_path}")

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