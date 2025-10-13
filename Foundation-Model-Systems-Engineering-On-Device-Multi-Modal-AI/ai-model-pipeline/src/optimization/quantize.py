# ai-model-pipeline/src/optimization/quantize.py

import torch
import torch.nn as nn
from transformers import BlipForConditionalGeneration, AutoProcessor
import yaml
from pathlib import Path
import os

def quantize_model(config: dict):
    """
    Loads a fine-tuned PyTorch model, applies dynamic quantization,
    and saves the quantized model.
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Quantization will run on CPU.")
    
    # 1. Define Paths
    # We load the fine-tuned model you uploaded to Hugging Face
    fine_tuned_model_name = config['model']['fine_tuned_hf_repo_id']
    quantized_model_path = Path(config['model']['quantized_path'])
    
    # Ensure the output directory exists
    quantized_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading fine-tuned model from: {fine_tuned_model_name}")

    # 2. Load the Fine-Tuned Model (on CPU for quantization)
    model = BlipForConditionalGeneration.from_pretrained(fine_tuned_model_name)
    model.to("cpu")
    model.eval() # Set the model to evaluation mode

    # 3. Apply Post-Training Dynamic Quantization
    # This converts the weights of linear layers from float32 to int8.
    print("Applying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear}, # Specify the layer types to quantize
        dtype=torch.qint8
    )
    print(" Model successfully quantized.")

    # 4. Save the Quantized Model's State Dictionary
    torch.save(quantized_model.state_dict(), quantized_model_path)
    print(f" Quantized model saved to: {quantized_model_path}")

    # 5. Compare File Sizes to Show Impact
    original_size = os.path.getsize(config['model']['fine_tuned_path']) / (1024*1024) # Placeholder path
    quantized_size = quantized_model_path.stat().st_size / (1024*1024)
    print("\n--- Optimization Impact ---")
    print(f"Original model size (approx): {original_size:.2f} MB") # This will be inaccurate until we download
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size / original_size) * 100:.2f}%")
    print("-------------------------")


if __name__ == '__main__':
    # Load configuration
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "configs" / "training_config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Add your Hugging Face repo ID to the config before running
    # For example:
    # config['model']['fine_tuned_hf_repo_id'] = "AshwinBalaji007/ashwin-blip-finetuned-oxford-pets"
    
    if 'fine_tuned_hf_repo_id' not in config['model']:
        raise ValueError("Please add 'fine_tuned_hf_repo_id' to your config.yaml")

    quantize_model(config)