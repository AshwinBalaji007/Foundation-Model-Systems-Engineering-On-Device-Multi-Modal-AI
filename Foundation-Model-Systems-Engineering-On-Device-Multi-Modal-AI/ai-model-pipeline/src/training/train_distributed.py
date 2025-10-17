# ======================================================================
# AI Systems Engineering: Distributed Training Script (Fixed Version)
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
from torch.cuda.amp import GradScaler  # use recommended GradScaler init below
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from data.dataset import get_dataloader

import functools
import warnings

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
    # last resort: pick token id 0 (rare but deterministic)
    warnings.warn("No usual special token ids found — falling back to token id 0 for decoder_start_token_id.")
    return 0

def setup():
    """Initializes the distributed process group and sets CUDA device."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

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

    # Ensure pad token exists (many tokenizers for GPT2 don't have pad_token)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model onto CPU, request dtype if provided
    load_dtype = getattr(torch, config['model'].get('dtype', 'bfloat16')) if 'dtype' in config['model'] else None
    model_kwargs = {}
    if load_dtype is not None:
        model_kwargs["torch_dtype"] = load_dtype  # transformers still accepts torch_dtype in many versions
    model = VisionEncoderDecoderModel.from_pretrained(model_name, **model_kwargs)

    # Ensure pad token id is set in both tokenizer and model config
    if tokenizer.pad_token_id is None and tokenizer.pad_token is not None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if tokenizer.pad_token_id is None:
        # As a final fallback, set pad token to eos or 0
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    model.config.pad_token_id = tokenizer.pad_token_id

    # Robustly set decoder_start_token_id (was causing the crash)
    chosen_decoder_start = choose_decoder_start_token_id(tokenizer)
    model.config.decoder_start_token_id = chosen_decoder_start

    # If model config requires vocab size match, ensure it's correct (optional)
    if hasattr(tokenizer, "vocab_size"):
        model.config.vocab_size = getattr(tokenizer, "vocab_size", model.config.vocab_size)

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

    if rank == 0:
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

    # Use recommended GradScaler init (device='cuda' is recommended in the FutureWarning)
    scaler = GradScaler()

    if rank == 0:
        print("\n--- Starting Training Loop with Stable Mixed Precision ---")
    model.train()

    # Prefer explicit device in autocast per newer API:
    autocast_ctx = torch.amp.autocast  # use in with-autocast(device_type='cuda')
    for epoch in range(config['training']['num_epochs']):
        # If your sampler supports set_epoch (DistributedSampler), set it
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        progress_bar = tqdm(dataloader, disable=(rank != 0), desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            # many vision-encoder-decoder examples expect labels = input_ids for teacher forcing
            if "input_ids" in batch:
                batch["labels"] = batch["input_ids"]

            # move tensors to device
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
        # sync between ranks at epoch end
        dist.barrier()

    if rank == 0:
        print("\n--- Training Complete ---")
        save_path = config['model']['fine_tuned_path']
        os.makedirs(save_path, exist_ok=True)
        # Proper FSDP saving is more involved; we print path as placeholder.
        print(f"✅ Model checkpoint would be saved to: {save_path}")

    cleanup()

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
