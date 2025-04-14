# models/base_vlm.py

from abc import ABC, abstractmethod
from PIL import Image
import torch
import base64 # Needed for vLLM image prep
import io       # Needed for vLLM image prep

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
        self.hf_device = device # Store HF device preference
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
