# src/models/llava_handler.py
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from typing import List, Dict, Any
import logging

from .base_vlm import BaseVLMHandler

logger = logging.getLogger(__name__)

class LlavaHandler(BaseVLMHandler):
    """Handles loading and interaction with LLaVA models."""

    def _load_model_and_processor(self):
        """Loads the LLaVA model and processor."""
        logger.info(f"Loading LLaVA model: {self.model_id}")
        try:
            # Note: LLaVA might use LlavaProcessor or AutoProcessor depending on version/variant
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
                quantization_config=self.quantization_config, # Pass resolved quantization config
                # attn_implementation="flash_attention_2" # Optional: if supported and beneficial
            )
            logger.info(f"LLaVA model {self.model_id} loaded successfully on device map: {self.model.hf_device_map}")
        except Exception as e:
            logger.error(f"Failed to load LLaVA model {self.model_id}: {e}")
            raise

    # def format_prompt(self, text_parts: List[str], images: List[Image.Image]) -> str:
    #     """
    #     Formats the prompt for LLaVA. LLaVA typically uses a simpler structure
    #     often involving USER:/ASSISTANT: and specific image tokens like <image>.
    #     The processor usually handles image token insertion.
    #     """
    #     if not self.processor:
    #         raise RuntimeError("Processor not loaded.")

    #     # LLaVA structure often looks like: "USER: <image>\n<text>\nASSISTANT:"
    #     # The processor inserts the image features where <image> token appears in text
    #     prompt_text = ""
    #     img_count = 0
    #     for part in text_parts:
    #         if part == "<image>":
    #              prompt_text += self.processor.tokenizer.decode(self.processor.tokenizer.convert_tokens_to_ids(['<image>'])[0]) # Use actual image token
    #              img_count += 1
    #         else:
    #             prompt_text += part

    #     # Add standard LLaVA turn structure if not already included in text_parts
    #     if not prompt_text.strip().startswith("USER:") and not prompt_text.strip().endswith("ASSISTANT:"):
    #          prompt_text = f"USER: {prompt_text}\nASSISTANT:"
    #     elif not prompt_text.strip().endswith("ASSISTANT:"):
    #          prompt_text += "\nASSISTANT:"


    #     if img_count != len(images):
    #          logger.warning(f"Number of images ({len(images)}) does not match <image> placeholders ({img_count}) in prompt parts.")

    #     # Unlike Qwen's apply_chat_template, for Llava we often pass the images
    #     # alongside the text prompt directly to the processor during the call.
    #     # So here we just return the text part of the prompt.
    #     return prompt_text.strip() # Return the text prompt string

    def generate_response(self, prompt: str, images: List[Image.Image], generation_args: Dict | None = None) -> str:
        """Generates a response from the LLaVA VLM."""
        if not self.model or not self.processor:
            raise RuntimeError("Model or processor not loaded.")

        default_args = {"max_new_tokens": 512, "do_sample": False}
        if generation_args:
            default_args.update(generation_args)

        # LLaVA processor typically takes text and images separately
        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            padding=True # Ensure padding if batching later
        ).to(self.model.device)

        logger.debug(f"Generating response with args: {default_args}")
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **default_args)

        # Remove input tokens from the generated output
        input_token_len = inputs.input_ids.shape[1]
        # Handle potential variations in output_ids shape if generation includes prompt
        if output_ids.shape[1] > input_token_len:
             generated_ids = output_ids[:, input_token_len:]
        else:
             # If generate only returns new tokens (check model config/behaviour)
             generated_ids = output_ids

        # Decode only the generated part
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        logger.debug(f"Raw VLM Response: {response[0]}")
        return response[0].strip()

    # Potentially override get_likelihood here if a LLaVA-specific method is found.