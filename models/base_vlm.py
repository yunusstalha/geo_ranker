# models/base_vlm.py

from abc import ABC, abstractmethod
from PIL import Image
import torch
import base64 # Needed for vLLM image prep
import io       # Needed for vLLM image prep
from typing import List, Dict, Union # Added typing

class BaseVLM(ABC):
    """
    Abstract base class for Vision-Language Models used in cross-view reranking.
    Ensures a common interface for Qwen, LLaVA, or any other VLM.
    """
    def __init__(self, model_name: str, device: str = "auto", inference_backend: str = 'hf', **kwargs):
        """
        Basic initializer to store model name and device.
        kwargs can be used for model-specific loading options.
        """
        self.model_name = model_name
        self.device = device

        if device == "auto":
            self.hf_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.hf_device = device # Respect user choice ('cuda', 'cpu')

        self.inference_backend = inference_backend
        self.kwargs = kwargs # Store extra args like tensor_parallel_size

        # Backend-specific attributes
        self.model = None # Will hold HF model or vLLM engine
        self.hf_processor = None # Store HF processor, potentially used by both

        self._load_model() # Trigger model loading


    @abstractmethod
    def _load_model(self, **kwargs):
        """
        Protected method to handle the actual loading of the model and processor.
        Subclasses must implement this.
        """
        pass
    

    def load_image(self, path: str) -> Image.Image:
        """
        Load an image from a path and return a PIL Image in RGB format.
        """
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def _prepare_vllm_image_input(self, pil_image: Image.Image) -> str:
        """ Helper to convert PIL image to base64 string for vLLM. """
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG") # Or JPEG
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str


    @abstractmethod
    def generate(self, conversation, image_inputs, max_new_tokens: int = 256) -> str:
        """
        Given a conversation (list/dict format) and a list of images,
        run inference and return the model's text output.
        """
        pass

    @abstractmethod
    def score_multiple_choice(self, conversation: Union[List, Dict], image_inputs: List[Image.Image], choices: List[str]) -> Dict[str, float]:
        """
        Scores a set of choices based on model's likelihood for the next token(s).
        Returns a dictionary mapping each choice string to its probability.
        Primarily intended for HF backend due to logit access requirements.
        """
        pass

    def move_inputs_to_device(self, inputs):
        """ Move inputs dictionary Tensors to the target device for HF. """
        if self.inference_backend == 'hf' and self.hf_device != 'cpu':
            target_device = torch.device(self.hf_device)
            for key in inputs.keys():
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(target_device)
        # No move needed for CPU or if device is handled by 'auto' map or vLLM
        return inputs