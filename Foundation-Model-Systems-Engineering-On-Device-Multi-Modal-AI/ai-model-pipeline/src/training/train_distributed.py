# ==============================================================================
# AI Systems Engineering: Distributed Training Script (Definitive Final Version)
# ==============================================================================
# This is the final, correct, and working version, including the professional
# logic for saving all model and processor components after training.

import sys
from pathlib import Path
src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

import os
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from data.dataset import get_dataloader
import functools

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
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    image_processor = ViTImageProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Critical model configuration for training
    tokenizer.pad_token = tokenizer.eos_token
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    if rank == 0: print("✅ Model successfully configured and wrapped with DDP.")

    if rank == 0: print("Creating distributed dataloaders...")
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
            batch["labels"] = batch["input_ids"]
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if rank == 0: progress_bar.set_postfix(loss=loss.item())
        dist.barrier()

    # --- THE DEFINITIVE, FINAL SAVING LOGIC ---
    if rank == 0:
        print("\n--- Training Complete ---")
        save_path = config['model']['fine_tuned_path']
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Saving model and processor components to: {save_path}")

        # `model.module` is used to access the original model inside the DDP wrapper.
        # `.save_pretrained()` saves the model weights, config, and generation config.
        model.module.save_pretrained(save_path)
        
        # We must also save the tokenizer...
        tokenizer.save_pretrained(save_path)
        
        # ...and the image processor to have a complete, reloadable artifact.
        image_processor.save_pretrained(save_path)
        
        print(f"✅ All components successfully saved.")

    cleanup()

# --- (The __main__ block remains the same) ---
if __name__ == '__main__':
    # ... (code is identical to the last working version) ...
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "configs" / "training_config.yaml"
    with open(config_path, 'r') as file: config = yaml.safe_load(file)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    train(rank, world_size, config)