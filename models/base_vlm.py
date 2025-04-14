# models/base_vlm.py

from abc import ABC, abstractmethod
from PIL import Image
import torch


class BaseVLM(ABC):
    """
    Abstract base class for Vision-Language Models used in cross-view reranking.
    Ensures a common interface for Qwen, LLaVA, or any other VLM.
    """
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        """
        Basic initializer to store model name and device.
        kwargs can be used for model-specific loading options.
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._load_model(**kwargs) # Trigger model loading in subclasses

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

    @abstractmethod
    def generate(self, conversation, image_inputs, max_new_tokens: int = 256) -> str:
        """
        Given a conversation (list/dict format) and a list of images,
        run inference and return the model's text output.
        """
        pass
