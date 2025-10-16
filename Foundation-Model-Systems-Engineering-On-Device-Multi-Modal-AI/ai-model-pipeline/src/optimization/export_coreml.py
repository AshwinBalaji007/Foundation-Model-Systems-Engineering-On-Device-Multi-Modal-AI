# ==============================================================================
# AI Systems Engineering: Core ML Conversion Script (to produce .mlmodel)
# ==============================================================================
#
# This version uses the `coremltools` library directly to gain fine-grained
# control over the conversion process. By specifying an older deployment target,
# we can force the output to be a single `.mlmodel` file instead of the newer
# `.mlpackage` directory format. This requires using `torch.jit.trace`.

import torch
import yaml
from pathlib import Path
import os
import sys
import coremltools as ct

# --- Add the 'src' directory to the Python path ---
src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))
from model.architecture import load_model_and_processor

def export_to_mlmodel(model: torch.nn.Module, processor, output_path: str):
    """
    Converts a PyTorch VisionEncoderDecoderModel to a single .mlmodel file
    by tracing its `forward` method and specifying a deployment target.
    """
    print("\n--- Starting Core ML Conversion (Targeting .mlmodel format) ---")
    model.eval()

    # --- Step 1: Create dummy inputs for tracing the `forward` pass ---
    # The tracer needs sample inputs to understand the model's graph.
    print("Creating dummy inputs for tracing...")
    dummy_pixel_values = torch.rand(1, 3, 224, 224) # ViT uses 224x224
    # The decoder needs a start token to begin generation.
    dummy_decoder_input_ids = torch.tensor([[processor.tokenizer.cls_token_id]], dtype=torch.long)

    # --- Step 2: Trace the model's `forward` method to TorchScript ---
    # This architecture is stable and compatible with tracing.
    print("Tracing the model's forward pass...")
    try:
        traced_model = torch.jit.trace(
            model, 
            (dummy_pixel_values, dummy_decoder_input_ids)
        )
    except Exception as e:
        print(f"❌ Tracing failed unexpectedly: {e}")
        return

    # --- Step 3: Convert the TorchScript Object to Core ML ---
    print("Converting TorchScript object to Core ML format...")
    coreml_model = ct.convert(
        traced_model,
        # --- THE DEFINITIVE FIX IS HERE ---
        # By specifying an older iOS target, we force the single .mlmodel format.
        minimum_deployment_target=ct.target.iOS14,
        
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="pixel_values", shape=dummy_pixel_values.shape),
            ct.TensorType(name="decoder_input_ids", shape=dummy_decoder_input_ids.shape, dtype=int),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    
    # --- Step 4: Add Metadata and Save ---
    coreml_model.author = "Ashwin Balaji"
    coreml_model.license = "MIT"
    coreml_model.short_description = "A fine-tuned ViT-GPT2 model for the Aura AI Assistant."
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    coreml_model.save(output_path)
    print(f"✅ Core ML model successfully converted and saved to: {output_path}")


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "configs" / "training_config.yaml"
    with open(config_path, 'r') as file: config = yaml.safe_load(file)

    fine_tuned_model_id = config['model']['fine_tuned_hub_id']
    coreml_output_path = config['model']['coreml_output_path']
    
    if not fine_tuned_model_id:
        print("❌ Error: 'fine_tuned_hub_id' not found in config.yaml.")
        sys.exit(1)

    print(f"--- Loading Fine-Tuned FP32 Model from Hub: {fine_tuned_model_id} ---")
    # Load the model in full float32 precision for a stable trace
    model, processor = load_model_and_processor(fine_tuned_model_id, dtype=torch.float32)
    print("✅ Fine-tuned FP32 model loaded successfully.")
    
    export_to_mlmodel(model, processor, coreml_output_path)