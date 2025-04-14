# src/models/loading.py
import logging
from .base_vlm import BaseVLMHandler
from .qwen_handler import QwenHandler
from .llava_handler import LlavaHandler

logger = logging.getLogger(__name__)

def get_vlm_handler(config: dict) -> BaseVLMHandler:
    """
    Factory function to instantiate the correct VLM handler based on config.

    Args:
        config (dict): The main configuration dictionary, expected to have
                       a 'model' key with 'name', 'model_id', etc.

    Returns:
        BaseVLMHandler: An instance of the appropriate VLM handler.

    Raises:
        ValueError: If the model name in the config is unsupported.
    """
    model_config = config.get("model", {})
    model_name = model_config.get("name", "").lower()
    model_id = model_config.get("model_id")
    device = model_config.get("device", "auto")
    torch_dtype = model_config.get("torch_dtype", "auto")
    quantization_config = model_config.get("quantization", {})


    if not model_id:
        raise ValueError("model.model_id must be specified in the config.")

    logger.info(f"Attempting to load VLM handler for model: {model_name} ({model_id})")

    if model_name == "qwen":
        return QwenHandler(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config
        )
    elif model_name == "llava":
        return LlavaHandler(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config
        )
    # Add elif branches here for other models like IDEFICS2 etc.
    # elif model_name == "idefics2":
    #     from .idefics2_handler import Idefics2Handler # Create this file
    #     return Idefics2Handler(...)
    else:
        raise ValueError(f"Unsupported VLM model name: {model_name}. "
                         f"Supported models: 'qwen', 'llava'.")