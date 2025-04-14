# models/qwen_vlm.py

import os
from PIL import Image
import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    # TorchAoConfig
    BitsAndBytesConfig
)
from .base_vlm import BaseVLM
import time

class QwenVLM(BaseVLM):
    def __init__(self, 
                model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                device: str = "auto",
                use_bf16: bool = True,
                use_quantization: bool = False # Control quantization
                ):
        """
        Initialize the QwenVLM model and processor.
        """
        self.use_bf16 = use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.use_quantization = use_quantization
        super().__init__(model_name=model_name, device=device) # This will call _load_model

    def _load_model(self):
        """
        Load the Qwen model and processor.
        Handles device mapping and data types.
        """
        print(f"Loading Qwen model: {self.model_name}...")
        start_time = time.time()

        quantization_config = None
        if self.use_quantization:
            print("Applying 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.use_bf16 else torch.float16
            )

        dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        if self.device == "cpu":
             dtype = torch.float32 # Override dtype for CPU
             print("Warning: Running on CPU, using float32. Performance will be slow.")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map=self.device if not self.use_quantization else "auto", # device_map='auto' often needed for quantization
                torch_dtype=dtype if not self.use_quantization else None, # dtype might conflict with quantization config
                quantization_config=quantization_config,
                trust_remote_code=True, # Important for models that require remote code execution
            )
            self.model.eval()

            # If device_map wasn't used (e.g., CPU), manually set device
            if self.device != "auto" and not self.model.hf_device_map:
                 self.model.to(self.device)

            print(f"Model loaded to device(s): {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else self.model.device}")

        except Exception as e:
            print(f"Error loading Qwen model {self.model_name}: {e}")
            print("Make sure you have `trust_remote_code=True` if required by the model.")
            raise

        end_time = time.time()
        print(f"Model loading took {end_time - start_time:.2f} seconds.")


# def generate(self, conversation, image_inputs, max_new_tokens: int = 256) -> str:
#     """
#     Generate a response for the given conversation and image inputs.
#     """
#     # Process the conversation and image inputs
#     self.processor.apply_chat_template(
#             conversation,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#     inputs = self.processor(
#         text=[prompt],
#         images=image_inputs,
#         padding=True,
#         return_tensors="pt"
#     )

#     output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

#     generated_ids = [
#         out_ids[len(in_ids):]
#         for in_ids, out_ids in zip(inputs.input_ids, output_ids)
#     ]

#     output_text = processor.batch_decode(
#         generated_ids,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=True
#     )

#     print("\nModel output:")
#     print(output_text[0])

    def generate(self, conversation: list, image_inputs: list[Image.Image], max_new_tokens: int = 256) -> str:
        """
        Generate a response for the given conversation and image inputs using Qwen format.
        """
        if not self.model or not self.processor:
             raise RuntimeError("Model and processor not loaded.")

        # 1. Apply chat template to the conversation (text parts only)
        #    The template should handle adding image placeholders like <image> if needed.
        #    We pass images separately to the processor.
        try:
            prompt = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True # Important for instructing the model to generate
            )
        except Exception as e:
             print(f"Error applying chat template: {e}")
             print("Conversation structure might be incompatible with the processor's template.")
             print(f"Conversation:\n{conversation}")
             raise

        # 2. Process text prompt and images together
        try:
            inputs = self.processor(
                text=[prompt], # Pass the templated prompt string as text
                images=image_inputs, # Pass the list of PIL images
                return_tensors="pt",
                padding=True # Pad if batching (here, batch size is 1)
            )
        except Exception as e:
            print(f"Error processing inputs with text and images: {e}")
            print(f"Prompt: {prompt}")
            print(f"Number of images: {len(image_inputs)}")
            raise

        # 3. Move inputs to the correct device(s)
        inputs = self.move_inputs_to_device(inputs)
        
        # 4. Generate
        # Use generate method of the loaded model
        # Need input_ids and pixel_values from the processor output
        output_ids = self.model.generate(**inputs, max_new_tokens=2048)
        
        # Trim the generated output if necessary.
        # We assume that the input prompt tokens are at the beginning.
        # Here we calculate the extra tokens generated.
        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False # Sometimes needed for specific models
        )[0] # Get the first (and only) item in the batch

        return output_text.strip()
        
    def move_inputs_to_device(self, inputs):
        """
        Move the inputs to the appropriate device(s).
        """
        if self.device == "auto":
            # If device is auto, let the processor handle it
            return inputs
        else:
            # Move each tensor in the inputs to the specified device
            for key in inputs.keys():
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
            return inputs