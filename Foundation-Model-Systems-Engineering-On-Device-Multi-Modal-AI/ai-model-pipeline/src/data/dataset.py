# ==============================================================================
# AI Systems Engineering: Professional Data Loading Module (Final Version)
# ==============================================================================
# This version includes the __len__ method, a requirement for all PyTorch datasets.

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor
import yaml
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
if not HUGGING_FACE_TOKEN:
    raise ValueError("Hugging Face token not found...")

class MultiModalDataset(Dataset):
    """Returns raw data points (PIL Image and text string)."""
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    # --- THE FIX IS HERE ---
    # The __len__ method is required by PyTorch's Dataset class.
    # It tells the DataLoader how many total samples are in the dataset,
    # which is essential for samplers and for creating progress bars.
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        try:
            class_id = item['label']
            text_content = f"a photo of a pet of class id {class_id}"
            image_content = item['image'].convert("RGB")
        except KeyError as e:
            raise KeyError(f"Dataset item at index {idx} is missing a required key: {e}.")
        return {"image": image_content, "text": text_content}

def create_collate_fn(processor):
    """Creates a collate function that uses the standard processor's batching."""
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        images = [item["image"] for item in batch]
        processed_batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        return processed_batch
    return collate_fn

def get_dataloader(rank: int, world_size: int, dataset_name: str, processor, batch_size: int, split: str = "train"):
    hf_dataset = load_dataset(dataset_name, split=split, token=HUGGING_FACE_TOKEN)
    pytorch_dataset = MultiModalDataset(hf_dataset)
    sampler = DistributedSampler(pytorch_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    collate_function = create_collate_fn(processor)
    dataloader = DataLoader(
        pytorch_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_function
    )
    return dataloader

# --- (The __main__ block remains the same) ---
if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "configs" / "training_config.yaml"
    with open(config_path, 'r') as file: config = yaml.safe_load(file)
    print("Loading processor for testing...")
    processor = AutoProcessor.from_pretrained(config['model']['base_model_name'], token=HUGGING_FACE_TOKEN)
    print(f"Creating a test dataloader for dataset: {config['data']['dataset_name']}...")
    test_dataloader = get_dataloader(
        rank=0, world_size=1, dataset_name=config['data']['dataset_name'], 
        processor=processor, batch_size=config['training']['batch_size_per_device']
    )
    print("\nFetching one batch from the dataloader...")
    try:
        for batch in test_dataloader:
            print("\n✅ Successfully fetched one batch.")
            print("Batch keys:", batch.keys())
            print("Input IDs shape:", batch['input_ids'].shape)
            print("Pixel values shape:", batch['pixel_values'].shape)
            break
    except Exception as e:
        print(f"\n❌ Error fetching batch: {e}")