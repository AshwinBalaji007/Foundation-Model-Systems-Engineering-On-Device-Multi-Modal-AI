# ai-model-pipeline/src/model/architecture.py
from transformers import BlipForConditionalGeneration, AutoProcessor

def load_model_and_processor(model_name_or_path: str, dtype=None):
    """
    Loads a BlipForConditionalGeneration model and its associated processor.
    This centralized function ensures consistency across the pipeline.

    Args:
        model_name_or_path (str): The name of the model on the Hugging Face Hub
                                  or the local path to a saved checkpoint.
        dtype: The torch data type to load the model in (e.g., torch.float16).

    Returns:
        A tuple of (model, processor).
    """
    model = BlipForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype
    )
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    
    print(f"Successfully loaded model and processor from: {model_name_or_path}")
    return model, processor