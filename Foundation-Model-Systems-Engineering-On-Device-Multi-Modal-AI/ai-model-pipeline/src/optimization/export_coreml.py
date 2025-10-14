# ==============================================================================
# AI Systems Engineering: Core ML Conversion Script (Memory-Efficient Version)
# ==============================================================================
import torch
import yaml
from pathlib import Path
import os
import sys
import coremltools as ct

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))
from model.architecture import load_model_and_processor

def export_to_coreml(model: torch.nn.Module, processor, output_path: str):
    print("\n--- Starting Core ML Conversion ---")
    model.eval()

    print("Creating dummy inputs for tracing...")
    dummy_text = ["a photo of a pet"]
    # Create dummy inputs that match the model's expected input dimensions
    dummy_pixel_values = torch.rand(1, 3, 384, 384)
    dummy_input_ids = torch.randint(0, 1000, (1, 32)) # Batch size 1, sequence length 32
    
    # We need to trace the model's forward method directly
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, pixel_values, input_ids):
            # We only trace the 'generate' method which is what we'll use for inference
            return self.model.generate(pixel_values=pixel_values, input_ids=input_ids)

    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()

    print("Tracing the model's generate function...")
    # Trace the wrapped model with the dummy inputs
    traced_model = torch.jit.trace(wrapped_model, (dummy_pixel_values, dummy_input_ids))

    print("Converting traced model to Core ML format...")
    coreml_model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="pixel_values", shape=dummy_pixel_values.shape),
            ct.TensorType(name="input_ids", shape=dummy_input_ids.shape),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    
    coreml_model.author = "Ashwin Balaji"
    coreml_model.license = "MIT"
    coreml_model.short_description = "A fine-tuned and quantized BLIP model for the Aura AI Assistant."

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    coreml_model.save(output_path)
    print(f"✅ Core ML model successfully converted and saved to: {output_path}")


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "configs" / "training_config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    base_model_id = config['model']['base_model_name']
    quantized_model_path = config['model']['quantized_path']
    coreml_output_path = config['model']['coreml_output_path']

    # --- THE MEMORY-EFFICIENT FIX IS HERE ---

    # 1. Load the BASE model architecture and processor
    print("--- Loading Base Model Architecture ---")
    model, processor = load_model_and_processor(base_model_id, dtype=None)

    # 2. Apply dynamic quantization to the architecture IN-PLACE
    print("Applying quantization to the model structure...")
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # 3. Load the saved quantized weights INTO the quantized architecture
    print(f"Loading quantized weights from: {quantized_model_path}")
    if not os.path.exists(quantized_model_path):
        print("❌ Error: Quantized model not found.")
        sys.exit(1)
        
    quantized_model.load_state_dict(torch.load(quantized_model_path))
    print("✅ Quantized weights loaded successfully.")
    
    # We now only have ONE model in memory: the quantized one.
    
    # 4. Run the Core ML conversion process
    export_to_coreml(quantized_model, processor, coreml_output_path)