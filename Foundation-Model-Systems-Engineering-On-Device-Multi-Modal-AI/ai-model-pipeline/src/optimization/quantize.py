# ==============================================================================
# AI Systems Engineering: Model Quantization Script
# ==============================================================================
#
# This script is responsible for the first stage of model optimization. It loads a
# fine-tuned model from a central repository (the Hugging Face Hub), applies
# post-training dynamic quantization to reduce its size and prepare it for
# efficient inference, and saves the resulting quantized model locally.

import torch
import yaml
from pathlib import Path
import os
import sys

# --- Add the 'src' directory to the Python path ---
# This allows us to import our custom modules like 'architecture.py'
src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

# Import our custom model loader from the 'model' module
from model.architecture import load_model_and_processor

def quantize_model(model: torch.nn.Module, output_path: str):
    """
    Applies post-training dynamic quantization to a PyTorch model.
    This is a highly effective technique for reducing model size (by ~4x)
    and accelerating CPU/on-device inference with minimal accuracy loss.

    Args:
        model (torch.nn.Module): The PyTorch model to be quantized.
        output_path (str): The local path where the quantized model's
                           state dictionary will be saved.
    """
    print("\n--- Starting Model Quantization ---")
    
    # Set the model to evaluation mode. This is crucial as it disables
    # layers like dropout that behave differently during training.
    model.eval()
    
    # Apply dynamic quantization. This function iterates through the model
    # and replaces the weights of specified layer types (here, torch.nn.Linear)
    # with their 8-bit integer equivalents.
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear},  # Specify the layer types to target for quantization
        dtype=torch.qint8   # The target data type for the quantized weights
    )
    
    print(" Model successfully quantized.")
    
    # Ensure the output directory exists before saving.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the state dictionary of the quantized model. This file contains
    # all the learned weights in their new, compressed format.
    torch.save(quantized_model.state_dict(), output_path)
    print(f" Quantized model state_dict saved to: {output_path}")

# This block is executed when the script is run directly from the command line.
if __name__ == '__main__':
    # --- 1. Load Project Configuration ---
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "configs" / "training_config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # --- 2. Get Model and Output Paths from Config ---
    # This is the professional, cloud-native workflow. We are loading our
    # fine-tuned model directly from its permanent home on the Hugging Face Hub.
    fine_tuned_model_id = config['model']['fine_tuned_hub_id']
    quantized_output_path = config['model']['quantized_path']
    
    # Validate that the Hub ID is present in the config
    if not fine_tuned_model_id:
        print("Error: 'fine_tuned_hub_id' not found in config.yaml.")
        print("Please specify the Hugging Face Hub repository ID for the fine-tuned model.")
        sys.exit(1)

    print(f"--- Loading Fine-Tuned Model from Hugging Face Hub ---")
    print(f"Model ID: {fine_tuned_model_id}")
        
    # --- 3. Load the Fine-Tuned Model ---
    # We load the model in full float32 precision (`dtype=None`) because the
    # quantization process requires the original, high-precision weights.
    model, _ = load_model_and_processor(fine_tuned_model_id, dtype=None)
    
    # --- 4. Run the Quantization Process ---
    quantize_model(model, quantized_output_path)