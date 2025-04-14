# src/models/qwen_handler.py
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
from typing import List, Dict, Any
import logging

from .base_vlm import BaseVLMHandler

logger = logging.getLogger(__name__)

class QwenHandler(BaseVLMHandler):
    """Handles loading and interaction with Qwen VL models."""

    def _load_model_and_processor(self):
        """Loads the Qwen model and processor."""
        logger.info(f"Loading Qwen model: {self.model_id}")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
                quantization_config=self.quantization_config, # Pass resolved quantization config
                trust_remote_code=True
            )
            logger.info(f"Qwen model {self.model_id} loaded successfully on device map: {self.model.hf_device_map}")
        except Exception as e:
            logger.error(f"Failed to load Qwen model {self.model_id}: {e}")
            raise

    def format_prompt(self, text_parts: List[str], images: List[Image.Image]) -> str:
        """
        Formats the prompt for Qwen using its chat template structure.
        Assumes image placeholders correspond sequentially to the images list.
        """
        if not self.processor:
            raise RuntimeError("Processor not loaded.")

        conversation = [{"role": "user", "content": []}]
        img_idx = 0
        for part in text_parts:
            if part == "<image>":
                if img_idx < len(images):
                    # Qwen's template typically handles images implicitly when passed to processor
                    # We add a placeholder text for image position in the conversation structure
                    conversation[0]["content"].append({"type": "image"})
                    img_idx += 1
                else:
                    logger.warning("More <image> placeholders than images provided.")
            else:
                conversation[0]["content"].append({"type": "text", "text": part})

        # Check if all images were used
        if img_idx != len(images):
             logger.warning(f"Number of images ({len(images)}) does not match <image> placeholders ({img_idx}).")

        try:
            # Tokenize=False gets the formatted string including special tokens for images
            prompt = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            logger.error(f"Error applying chat template: {e}")
            # Fallback or simplified formatting if template fails
            simple_prompt = ""
            for part in conversation[0]["content"]:
                if part["type"] == "text":
                    simple_prompt += part["text"] + "\n"
                elif part["type"] == "image":
                    simple_prompt += "<image placeholder>\n" # Indicate image position
            logger.warning("Using simplified prompt formatting due to template error.")
            return simple_prompt


    def generate_response(self, prompt: str, images: List[Image.Image], generation_args: Dict | None = None) -> str:
        """Generates a response from the Qwen VLM."""
        if not self.model or not self.processor:
            raise RuntimeError("Model or processor not loaded.")

        default_args = {"max_new_tokens": 512, "do_sample": False} # Sensible defaults
        if generation_args:
            default_args.update(generation_args)

        inputs = self.processor(
            text=[prompt],
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device) # Ensure inputs are on the correct device

        logger.debug(f"Generating response with args: {default_args}")
        with torch.no_grad():
             # Use **inputs.to_dict() if error occurs
            output_ids = self.model.generate(**inputs, **default_args)

        # Remove input tokens from the generated output
        input_token_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[:, input_token_len:]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        logger.debug(f"Raw VLM Response: {response[0]}")
        return response[0].strip()

    # Potentially override get_likelihood here if a Qwen-specific method is found
    # e.g., using logits or specific API features if available.