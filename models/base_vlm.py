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
    def format_prompt(self, text_parts: List[str], images: List[Image.Image]) -> str:
        """
        Formats the text parts and image placeholders into a single prompt string
        suitable for the specific VLM's chat template.

        Args:
            text_parts (List[str]): List of text segments for the prompt.
            images (List[Image.Image]): List of images corresponding to placeholders in the prompt.

        Returns:
            str: The fully formatted prompt string.
        """
        pass

    @abstractmethod
    def generate_response(self, prompt: str, images: List[Image.Image], generation_args: Dict | None = None) -> str:
        """
        Generates a response from the VLM given a prompt and images.

        Args:
            prompt (str): The formatted prompt string.
            images (List[Image.Image]): The list of images to be processed.
            generation_args (Dict | None): Optional arguments for the model's generate method
                                           (e.g., max_new_tokens, temperature).

        Returns:
            str: The generated text response from the VLM.
        """
        pass

    def get_likelihood(self, prompt: str, images: List[Image.Image], target_texts: List[str]) -> List[float]:
        """
        Calculates the likelihood (e.g., negative log probability) of generating
        specific target texts given the prompt and images.
        (This is more complex and might require accessing logits - basic implementation provided,
        but may need model-specific refinement).

        Args:
            prompt (str): The formatted prompt string.
            images (List[Image.Image]): The list of images.
            target_texts (List[str]): The candidate texts whose likelihoods are to be calculated.

        Returns:
            List[float]: A list of scores (e.g., negative log-likelihoods) for each target text.
                         Lower scores usually indicate higher likelihood.
        """
        # NOTE: This is a simplified placeholder. Accurate likelihood calculation often
        # requires accessing token logits and careful handling of tokenization.
        # May need significant model-specific adaptation.
        print("Warning: get_likelihood is a basic placeholder and may not be accurate.")

        inputs = self.processor(text=[prompt] * len(target_texts), images=images * len(target_texts), return_tensors="pt", padding=True).to(self.model.device)
        target_tokens = self.processor(text=target_texts, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.to(self.model.device)

        outputs = self.model(**inputs, output_hidden_states=False, output_attentions=False)
        logits = outputs.logits

        # Simplified score: Sum log probabilities of target tokens (needs careful alignment)
        # This requires aligning input_ids+target_ids with logits correctly.
        # A common simplification is to just use the generate function and hope the first token
        # or a simple generated score reflects likelihood, which is often inaccurate.
        # Returning dummy scores for now.
        return [-float(i) for i in range(len(target_texts))] # Return dummy decreasing scores