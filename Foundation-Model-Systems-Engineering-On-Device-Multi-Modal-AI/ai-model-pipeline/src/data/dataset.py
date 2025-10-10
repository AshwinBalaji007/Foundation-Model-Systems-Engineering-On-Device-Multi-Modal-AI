# ==============================================================================
# AI Systems Engineering: Professional Data Loading Module (CIFAR10 Fallback)
# ==============================================================================
#
# This script uses the built-in torchvision CIFAR10 dataset.
# It is simpler and has no external dependencies, making it a reliable way
# to test the distributed training pipeline.

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml
from pathlib import Path

def get_dataloader(rank: int, world_size: int, batch_size: int, data_path: str = "./data"):
    """
    Creates a DataLoader for the CIFAR10 dataset with a DistributedSampler.
    """
    # Define a standard transformation for image normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the training dataset
    dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
    return dataloader

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "configs" / "training_config.yaml"
    with open(config_path, 'r') as file: config = yaml.safe_load(file)

    print("Creating a test dataloader for CIFAR10...")
    test_dataloader = get_dataloader(
        rank=0, 
        world_size=1, 
        batch_size=config['training']['batch_size_per_device']
    )
    
    print("\nFetching one batch from the dataloader...")
    try:
        # The CIFAR10 dataloader returns a tuple (images, labels)
        for images, labels in test_dataloader:
            print("\n✅ Successfully fetched one batch.")
            print("Images batch shape:", images.shape)
            print("Labels batch shape:", labels.shape)
            break
    except Exception as e:
        print(f"\n❌ Error fetching batch: {e}")