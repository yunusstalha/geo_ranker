# models/base_vlm.py

from abc import ABC, abstractmethod
from PIL import Image
import torch


class BaseVLM(ABC):
    """
    Abstract base class for Vision-Language Models used in cross-view reranking.
    Ensures a common interface for Qwen, LLaVA, or any other VLM.
    """

    @abstractmethod
    def load_image(self, path: str) -> Image.Image:
        """
        Load an image from a path and return a PIL Image in RGB format.
        """
        pass


    @abstractmethod
    def generate(self, conversation, image_inputs, max_new_tokens: int = 256) -> str:
        """
        Given a conversation (list/dict format) and a list of images,
        run inference and return the model's text output.
        """
        pass
