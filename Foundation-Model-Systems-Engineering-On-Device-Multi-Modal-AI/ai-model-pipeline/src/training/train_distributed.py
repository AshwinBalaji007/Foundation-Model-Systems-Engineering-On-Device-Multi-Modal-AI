# ==============================================================================
# AI Systems Engineering: Distributed Training Script (Phi-3 Optimized)
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
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
# --- THE CHANGE IS HERE ---
from transformers import AutoProcessor, AutoModelForCausalLM # Phi-3 Vision uses the CausalLM class
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
    
    if rank == 0:
        print(f"Loading base model '{model_name}' onto CPU...")

    # --- THE CHANGES ARE HERE for Phi-3 ---
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Phi-3 is a powerful model that requires you to trust its code.
    # It uses the AutoModelForCausalLM class.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True 
    )

    fsdp_sharding_strategy = ShardingStrategy.FULL_SHARD
    cpu_offload = CPUOffload(offload_params=True)
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e7) # Lowering for a smaller model
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=fsdp_sharding_strategy,
        cpu_offload=cpu_offload,
        device_id=torch.cuda.current_device()
    )
    
    if rank == 0:
        print("✅ Model successfully wrapped with FSDP using FULL_SHARD and CPU Offload strategy.")

    if rank == 0:
        print("Creating distributed dataloaders...")
    dataloader = get_dataloader(
        rank=rank,
        world_size=world_size,
        dataset_name=config['data']['dataset_name'],
        processor=processor,
        batch_size=config['training']['batch_size_per_device']
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    if rank == 0:
        print("\n--- Starting Training Loop ---")

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
            
            if rank == 0:
                progress_bar.set_postfix(loss=loss.item())

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
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires at least one CUDA-enabled GPU.")
    if world_size > torch.cuda.device_count():
         raise RuntimeError(f"Requested {world_size} GPUs, but only {torch.cuda.device_count()} are available.")
    train(rank, world_size, config)