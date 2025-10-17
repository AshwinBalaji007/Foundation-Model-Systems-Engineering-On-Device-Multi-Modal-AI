# ======================================================================
# AI Systems Engineering: Distributed Training Script (Fixed + tqdm Single Line)
# ======================================================================

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
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from data.dataset import get_dataloader

import functools
import warnings


# -----------------------------
# Utility Functions
# -----------------------------
def choose_decoder_start_token_id(tokenizer):
    """Pick a sensible decoder_start_token_id with fallbacks."""
    candidates = [
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "cls_token_id", None),
        getattr(tokenizer, "sep_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    ]
    for c in candidates:
        if c is not None:
            return c
    warnings.warn("No usual special token ids found — falling back to token id 0 for decoder_start_token_id.")
    return 0


def _map_dtype_string_to_torch(dtype_str):
    if not dtype_str:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_str, None)


def setup():
    """Initializes the distributed process group and sets CUDA device."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    """Cleans up the distributed process group."""
    try:
        dist.destroy_process_group()
    except Exception:
        pass


# -----------------------------
# Main Training Loop
# -----------------------------
def train(rank: int, world_size: int, config: dict):
    if rank == 0:
        print("--- Initializing Training on All Ranks using FSDP ---")

    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    model_name = config['model']['base_model_name']
    if rank == 0:
        print(f"Loading base model '{model_name}' onto CPU...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    image_processor = ViTImageProcessor.from_pretrained(model_name)

    # Ensure pad token exists
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Map dtype
    dtype_str = config.get('model', {}).get('dtype', None)
    torch_dtype_obj = _map_dtype_string_to_torch(dtype_str)
    model_load_kwargs = {}
    if torch_dtype_obj is not None:
        model_load_kwargs["torch_dtype"] = torch_dtype_obj

    model = VisionEncoderDecoderModel.from_pretrained(model_name, **model_load_kwargs)

    # Fix pad and decoder tokens
    if tokenizer.pad_token_id is None and tokenizer.pad_token is not None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    model.config.pad_token_id = tokenizer.pad_token_id

    chosen_decoder_start = choose_decoder_start_token_id(tokenizer)
    model.config.decoder_start_token_id = chosen_decoder_start

    if rank == 0:
        print(f"Using decoder_start_token_id = {model.config.decoder_start_token_id}")
        print("Wrapping model with FSDP...")

    fsdp_sharding_strategy = ShardingStrategy.FULL_SHARD
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e7)

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=fsdp_sharding_strategy,
        device_id=torch.cuda.current_device()
    )

    if rank == 0:
        print("✅ Model successfully configured and wrapped with FSDP.")
        print("Creating distributed dataloaders...")

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

    if rank == 0:
        print("\n--- Starting Training Loop with Stable Mixed Precision ---")

    model.train()
    autocast_ctx = torch.amp.autocast

    try:
        for epoch in range(config['training']['num_epochs']):
            if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)

            # ✅ tqdm single-line setup
            progress_bar = tqdm(
                dataloader,
                disable=(rank != 0),
                desc=f"Epoch {epoch+1}",
                dynamic_ncols=True,
                leave=False,
                position=0,
                file=sys.stdout
            )

            for batch in progress_bar:
                if "input_ids" in batch:
                    batch["labels"] = batch["input_ids"]

                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad()
                with autocast_ctx(device_type='cuda'):
                    outputs = model(**batch)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if rank == 0:
                    progress_bar.set_postfix(loss=float(loss.detach().cpu().item()))

            dist.barrier()  # sync between ranks at epoch end

        if rank == 0:
            print("\n--- Training Complete ---")
            save_path = config['model']['fine_tuned_path']
            os.makedirs(save_path, exist_ok=True)
            print(f"✅ Model checkpoint would be saved to: {save_path}")

    finally:
        cleanup()


# -----------------------------
# Entry Point
# -----------------------------
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
