# src/models/base_vlm.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from PIL import Image
import torch

class BaseVLMHandler(ABC):
    """Abstract Base Class for Vision-Language Model Handlers."""

    def __init__(self, model_id: str, device: str = "auto", torch_dtype: str = "auto", quantization_config: Dict | None = None):
        """
        Initializes the VLM handler.

        Args:
            model_id (str): The identifier for the pre-trained model.
            device (str): The device to load the model onto ('cuda', 'cpu', 'auto').
            torch_dtype (str): The desired torch dtype ('bfloat16', 'float16', 'float32', 'auto').
            quantization_config (Dict | None): Configuration for model quantization.
        """
        self.model_id = model_id
        self.device_map = device if device in ["auto", "cuda", "cpu"] else "auto" # Transformers device_map expects these
        self.torch_dtype_str = torch_dtype
        self.quantization_config_dict = quantization_config
        self.model = None
        self.processor = None
        self._resolve_dtype()
        self._resolve_quantization()
        self._load_model_and_processor()

    def _resolve_dtype(self):
        """Resolves the torch dtype string to a torch.dtype object."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto" # Keep as string for transformers auto-detection
        }
        self.torch_dtype = dtype_map.get(self.torch_dtype_str, "auto")
        if self.torch_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
             print("Warning: bfloat16 specified but not supported. Falling back to float16.")
             self.torch_dtype = torch.float16

    def _resolve_quantization(self):
        """Resolves the quantization configuration dictionary."""
        # Placeholder for specific quantization library setup if needed beyond passing the dict
        self.quantization_config = None # Default to None
        if self.quantization_config_dict and self.quantization_config_dict.get("enabled", False):
            mode = self.quantization_config_dict.get("mode")
            # Example: Handling TorchAO quantization config specifically
            if mode == "int4_weight_only":
                try:
                    from transformers import TorchAoConfig
                    group_size = self.quantization_config_dict.get("group_size", 128)
                    self.quantization_config = TorchAoConfig(mode, group_size=group_size)
                    print(f"Using TorchAoConfig quantization: mode={mode}, group_size={group_size}")
                except ImportError:
                    print("Warning: TorchAO not installed or config incorrect. Quantization disabled.")
            # Example: Handling bitsandbytes quantization (common for LLMs)
            elif mode in ["int8", "int4"]:
                 try:
                    from transformers import BitsAndBytesConfig
                    load_in_4bit = mode == "int4"
                    load_in_8bit = mode == "int8"
                    # Add other BitsAndBytes options here if needed from config
                    self.quantization_config = BitsAndBytesConfig(
                        load_in_4bit=load_in_4bit,
                        load_in_8bit=load_in_8bit,
                        # Example: bnb_4bit_compute_dtype=torch.float16,
                        # Example: bnb_4bit_use_double_quant=True,
                        # Example: bnb_4bit_quant_type="nf4"
                    )
                    print(f"Using BitsAndBytes quantization: load_in_4bit={load_in_4bit}, load_in_8bit={load_in_8bit}")
                 except ImportError:
                     print("Warning: bitsandbytes not installed or config incorrect. Quantization disabled.")
            else:
                print(f"Warning: Unsupported quantization mode '{mode}'. Quantization disabled.")


    @abstractmethod
    def _load_model_and_processor(self):
        """Loads the specific VLM model and processor."""
        pass


    @abstractmethod
    def generate_response(self,
                          prompt_structure: List[Dict[str, Any]],
                          images: List[Image.Image],
                          generation_args: Dict | None = None
                         ) -> str:
        """
        Generates a response from the VLM given a structured prompt and images.

        Args:
            prompt_structure (List[Dict[str, Any]]): The structured prompt representation
                (e.g., [{'type': 'text', 'content': '...'}, {'type': 'image', 'index': 0}]).
            images (List[Image.Image]): The list of images, indexed according to 'index'
                                       in the prompt_structure.
            generation_args (Dict | None): Optional arguments for the model's generate method
                                           (e.g., max_new_tokens, temperature).

        Returns:
            str: The generated text response from the VLM.
        """
        pass


    def get_likelihood(self,
                       prompt_structure: List[Dict[str, Any]],
                       images: List[Image.Image],
                       target_texts: List[str]
                      ) -> List[float]:
        """
        Calculates the likelihood of generating specific target texts.
        (Placeholder - requires significant model-specific implementation).

        Args:
            prompt_structure: The structured prompt representation.
            images: The list of images.
            target_texts: The candidate texts whose likelihoods are to be calculated.

        Returns:
            List[float]: A list of scores (e.g., negative log-likelihoods).
        """
        # This method now also needs to handle the prompt_structure to prepare inputs.
        # The core difficulty of accurate likelihood calculation remains.
        print("get_likelihood is a basic placeholder and may not be accurate or functional with structured prompts.")
        # Placeholder: Attempt to prepare inputs similarly to how generate_response might
        # This part is highly dependent on the specific model's processor
        try:
            # Attempt a generic preparation (likely needs override in subclass)
             prepared_inputs = self._prepare_inputs_for_generation(prompt_structure, images)
             inputs_list = [prepared_inputs] * len(target_texts) # Naive duplication for batching

             # Replicate processor call from generate response if possible (might need text only)
             # This is very difficult to generalize here. Subclasses MUST override.
             # inputs = self.processor(text=[prompt_text_from_structure] * len(target_texts), images=images * len(target_texts), return_tensors="pt", padding=True).to(self.model.device)

             print("get_likelihood needs model-specific implementation to prepare inputs from structure.")
             return [-float(i) for i in range(len(target_texts))] # Return dummy scores

        except Exception as e:
             print(f"Failed to prepare inputs for get_likelihood (needs override): {e}")
             return [-float(i) for i in range(len(target_texts))]