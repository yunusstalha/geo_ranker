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

    def _structure_to_qwen_conversation(self, prompt_structure: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Converts the generic prompt structure to Qwen's expected chat format.
        e.g., [{'role': 'user', 'content': [{'type':'text', 'text':'...'}, {'type':'image'}]}]
        """
        qwen_content = []
        for part in prompt_structure:
            part_type = part.get('type')
            if part_type == 'text':
                # Qwen expects 'text' key within content list
                qwen_content.append({'type': 'text', 'text': part.get('content', '')})
            elif part_type == 'image':
                # Qwen expects just 'image' type; index mapping happens implicitly later
                qwen_content.append({'type': 'image'})
            else:
                logger.warning(f"Unsupported part type '{part_type}' in prompt structure for Qwen. Skipping.")
        # Assuming a single user turn for now, adjust if multi-turn needed
        return [{"role": "user", "content": qwen_content}]


    def generate_response(self,
                        prompt_structure: List[Dict[str, Any]],
                        images: List[Image.Image],
                        generation_args: Dict | None = None
                        ) -> str:
        """Generates a response from the Qwen VLM using structured input."""
        if not self.model or not self.processor:
            # Add check here as well, in case loading failed silently
            logger.error("Model or processor not loaded. Cannot generate response.")
            raise RuntimeError("Model or processor not loaded.")

        logger.debug(f"Received prompt structure with {len(prompt_structure)} parts and {len(images)} images.")

        # 1. Convert structure to Qwen's chat format
        conversation = self._structure_to_qwen_conversation(prompt_structure)
        logger.debug(f"Converted structure to Qwen conversation: {conversation}")

        # 2. Apply chat template to get the final prompt string
        try:
            # Tokenize=False gets the formatted string including special tokens for images/roles
            prompt_string = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True # Important for generation
            )
            logger.debug(f"Formatted prompt string for processor: '{prompt_string[:200]}...'")
        except Exception as e:
            logger.error(f"Error applying Qwen chat template: {e}", exc_info=True)
            # Fallback: Try to concatenate text parts? Very crude.
            prompt_string = " ".join([p.get('content', '') for p in prompt_structure if p.get('type') == 'text'])
            logger.warning("Using simplified fallback prompt string due to template error.")


        # 3. Prepare model inputs using the processor
        try:
             # The processor takes the template string and the original image list.
             # It maps the image placeholders in the template string to the images.
             inputs = self.processor(
                 text=[prompt_string], # Pass the templated string
                 images=images,      # Pass the original list of images
                 padding=True,
                 return_tensors="pt"
             ).to(self.model.device) # Ensure inputs are on the correct device
             logger.debug("Processor created model inputs.")
        except Exception as e:
             logger.error(f"Error processing inputs with Qwen processor: {e}", exc_info=True)
             raise # Cannot proceed without inputs


        # 4. Run model generation (remains largely the same)
        default_args = {"max_new_tokens": 512, "do_sample": False} # Sensible defaults
        if generation_args:
            default_args.update(generation_args)

        logger.debug(f"Generating response with args: {default_args}")
        output_ids = None # Initialize
        try:
            with torch.no_grad():
                 output_ids = self.model.generate(**inputs, **default_args)
            logger.debug("Model generation completed.")
        except Exception as e:
             logger.error(f"Error during Qwen model.generate: {e}", exc_info=True)
             # Handle error, maybe return an error message or raise
             return "[ERROR: Generation failed]"

        if output_ids is None:
             logger.error("Output IDs are None after generation call.")
             return "[ERROR: Generation produced no output]"

        # 5. Decode response (remains largely the same)
        try:
            input_token_len = inputs.input_ids.shape[1]
            # Ensure slicing is correct even if output_ids is shorter than input (shouldn't happen with generate)
            generated_ids = output_ids[:, input_token_len:] if output_ids.shape[1] > input_token_len else output_ids

            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            final_response = response[0].strip() if response else ""
            logger.debug(f"Decoded VLM Response: {final_response}")
            return final_response
        except Exception as e:
             logger.error(f"Error decoding Qwen response: {e}", exc_info=True)
             return "[ERROR: Decoding failed]"

    # Potentially override get_likelihood here if a Qwen-specific method is found
    # e.g., using logits or specific API features if available.