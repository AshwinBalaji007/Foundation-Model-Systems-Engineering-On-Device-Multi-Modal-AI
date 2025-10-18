# ai-model-pipeline/src/model/architecture.py
import torch
from transformers import VisionEncoderDecoderModel, AutoProcessor
from pathlib import Path

def load_model_and_processor(model_id: str, dtype=torch.float32):
    """
    Loads a VisionEncoderDecoder model and its processor.

    Args:
        model_id (str): Hugging Face model hub ID of the fine-tuned ViT+GPT2 model
        dtype: torch data type (default: torch.float32)

    Returns:
        model: VisionEncoderDecoderModel (eval mode)
        processor: AutoProcessor for preprocessing images & text
    """
    if not model_id:
        raise ValueError("model_id cannot be empty!")

    # --- Load processor ---
    processor = AutoProcessor.from_pretrained(model_id)
    
    # --- Load the model ---
    # Use VisionEncoderDecoderModel because your fine-tuned model is ViT + GPT-2
    model = VisionEncoderDecoderModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    
    # Put model in eval mode
    model.eval()

    # Optional: disable gradient computations for export
    for param in model.parameters():
        param.requires_grad = False

    return model, processor


def save_dummy_model_artifacts(model_id: str, output_dir: str):
    """
    Optional helper to test loading and saving dummy artifacts.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor(model_id)

    # Save a dummy TorchScript of encoder
    dummy_input = torch.rand(1, 3, 224, 224)
    traced_encoder = torch.jit.trace(model.encoder, dummy_input)
    traced_encoder.save(output_path / "dummy_encoder.pt")

    print(f"✅ Dummy encoder saved at {output_path / 'dummy_encoder.pt'}")