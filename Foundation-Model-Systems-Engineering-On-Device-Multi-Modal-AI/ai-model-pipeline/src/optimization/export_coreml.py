# ai-model-pipeline/src/optimization/export_coreml.py
# ==============================================================================
# Hybrid On-Device Artifact Export Script (Vision → Core ML + GPT-2 PyTorch Mobile)
# ==============================================================================
import os
import sys
from pathlib import Path
import torch
import coremltools as ct
import yaml

# Add 'src' to Python path
src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))
from model.architecture import load_model_and_processor

if __name__ == "__main__":
    # Ensure macOS
    if sys.platform != "darwin":
        print("❌ Error: This script requires macOS.")
        sys.exit(1)

    # --- Load config ---
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "configs" / "training_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    fine_tuned_model_id = config["model"]["fine_tuned_hub_id"]
    output_dir = Path(config["model"]["coreml_output_path"]).parent
    os.makedirs(output_dir, exist_ok=True)

    if not fine_tuned_model_id or "YourHuggingFaceUsername" in fine_tuned_model_id:
        print("❌ Error: 'fine_tuned_hub_id' not set correctly in config.yaml")
        sys.exit(1)

    # --- Load fine-tuned VisionEncoderDecoder model ---
    print(f"--- Loading Fine-Tuned FP32 Model from Hub: {fine_tuned_model_id} ---")
    model, processor = load_model_and_processor(fine_tuned_model_id, dtype=torch.float32)
    model.eval()
    print("✅ Fine-tuned model loaded successfully.\n")

    # ======================================================================
    # Part 1: Export Vision Encoder to Core ML (ANE)
    # ======================================================================
    print("--- Part 1: Exporting Vision Encoder to Core ML ---")

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, pixel_values):
            outputs = self.encoder(pixel_values=pixel_values)
            return outputs.last_hidden_state

    encoder_wrapper = EncoderWrapper(model.encoder)
    dummy_pixel_values = torch.rand(1, 3, 224, 224)

    try:
        traced_encoder = torch.jit.trace(encoder_wrapper, dummy_pixel_values)
        coreml_encoder = ct.convert(
            traced_encoder,
            inputs=[ct.ImageType(name="image", shape=dummy_pixel_values.shape, scale=1/255.0)],
            minimum_deployment_target=ct.target.iOS17,
            compute_units=ct.ComputeUnit.CPU_AND_NE,  # Use ANE
        )
        encoder_output_path = output_dir / "VisionEncoder.mlpackage"
        coreml_encoder.save(str(encoder_output_path))
        print(f"✅ Vision Encoder saved to: {encoder_output_path}")
    except Exception as e:
        print(f"❌ Failed to convert Vision Encoder: {e}")

    # ======================================================================
    # Part 2: Save GPT-2 Decoder for PyTorch Mobile
    # ======================================================================
    print("\n--- Part 2: Saving GPT-2 Decoder for PyTorch Mobile ---")

    decoder_output_dir = output_dir / "TextDecoder"
    os.makedirs(decoder_output_dir, exist_ok=True)

    try:
        model.decoder.save_pretrained(decoder_output_dir)
        print(f"✅ GPT-2 Decoder saved to: {decoder_output_dir}")
        print("ℹ️ Load this decoder in PyTorch Mobile for on-device autoregressive generation.")
    except Exception as e:
        print(f"❌ Failed to save GPT-2 Decoder: {e}")

    print("\n--- Export complete ---")
