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
from dataclasses import asdict
try:
    from vllm import LLM, EngineArgs, SamplingParams
    VLLM_AVAILABLE = True
    from qwen_vl_utils import process_vision_info
    QWEN_UTILS_AVAILABLE = False
except ImportError:
    process_vision_info = None
    QWEN_UTILS_AVAILABLE = False
    print("Optional: `qwen-vl-utils` not found. Image resizing specific to Qwen might not occur.")

from .base_vlm import BaseVLM
import time

import base64
import io

class QwenVLM(BaseVLM):
    # Update __init__ signature
    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "auto",
                 use_bf16: bool = True,
                 use_quantization: bool = False,
                 inference_backend: str = 'hf', # 'hf' or 'vllm'
                 tensor_parallel_size: int = 1, # For vLLM
                 max_images_per_prompt: int = 5, # For vLLM
                ):
        self.use_bf16 = use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.use_quantization = use_quantization
        self.tensor_parallel_size = tensor_parallel_size
        self.max_images_per_prompt = max_images_per_prompt
        super().__init__(model_name=model_name, device=device, inference_backend=inference_backend)


    def _load_model(self):
        """ Load model based on the selected inference backend. """
        if self.inference_backend == 'hf':
            self._load_hf_model()
        elif self.inference_backend == 'vllm':
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM backend selected, but vLLM library is not installed or failed to import.")
            self._load_vllm_engine() # This might fail if Qwen-VL not supported
        else:
            raise ValueError(f"Unsupported inference_backend: {self.inference_backend}")
   
    def _load_hf_model(self):
        """ Loads the HuggingFace Qwen model (existing logic). """
        print(f"Loading Qwen model for HF backend: {self.model_name}...")
        start_time = time.time()
        quantization_config_bnb = None
        if self.use_quantization:
            print("Applying 4-bit BitsAndBytes quantization for HF...")
            quantization_config_bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.use_bf16 else torch.float16,
            )

        dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        effective_device_map = self.hf_device
        model_dtype_arg = dtype

        if self.hf_device == "cpu":
            model_dtype_arg = torch.float32
            effective_device_map = "cpu"
            print("Warning: Running HF on CPU, forcing float32.")
            if self.use_quantization:
                self.use_quantization = False
                quantization_config_bnb = None

        if self.use_quantization:
             effective_device_map = "auto"
             model_dtype_arg = None
             print(f"Using HF quantization. Setting device_map='{effective_device_map}' and model_dtype_arg=None.")

        try:
            self.hf_processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map=effective_device_map,
                torch_dtype=model_dtype_arg,
                quantization_config=quantization_config_bnb,
                trust_remote_code=True,
            )
            self.model.eval()
            print(f"HF Model loaded. Device map: {getattr(self.model, 'hf_device_map', 'N/A')}. Effective device: {self.model.device}")
        except Exception as e:
            print(f"Error loading Qwen HF model {self.model_name}: {e}")
            raise
        end_time = time.time()
        print(f"HF Model loading took {end_time - start_time:.2f} seconds.")


    def _load_vllm_engine(self):
        """ Loads the vLLM engine for Qwen. """
        print(f"Loading Qwen model for vLLM backend: {self.model_name}...")
        start_time = time.time()

        print("Loading HF Processor for text processing...")
        try:
            self.hf_processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Could not load HF Processor {self.model_name}: {e}")
            self.hf_processor = None

        vllm_quantization = None
        if self.use_quantization:
            vllm_quantization = "awq"
            print(f"Applying vLLM quantization: {vllm_quantization}")

        max_images_per_prompt = self.max_images_per_prompt
        engine_args = EngineArgs(
            model=self.model_name,
            quantization=vllm_quantization,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": max_images_per_prompt},
        )
        print(f"vLLM EngineArgs: {engine_args}")

        try:
             # *** FIX: Use asdict() instead of .to_dict() ***
            self.model = LLM(**asdict(engine_args))
            print("vLLM Engine loaded successfully for Qwen.")
        except Exception as e:
            print(f"Error loading vLLM engine for Qwen model {self.model_name}: {e}")
            raise

        end_time = time.time()
        print(f"vLLM Engine loading took {end_time - start_time:.2f} seconds.")

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
        """ Generate text using the selected backend. """
        if self.inference_backend == 'hf':
            return self._generate_hf(conversation, image_inputs, max_new_tokens)
        elif self.inference_backend == 'vllm':
            # Note: This part might not work if Qwen-VL is not supported by vLLM
            return self._generate_vllm(conversation, image_inputs, max_new_tokens)
        else:
            raise ValueError(f"Unsupported inference_backend: {self.inference_backend}")

    def _generate_hf(self, conversation: list, image_inputs: list[Image.Image], max_new_tokens: int) -> str:
        """ Generate using HuggingFace backend (existing logic). """
        if not self.model or not self.hf_processor:
            raise RuntimeError("HF Model or processor not loaded for Qwen HF generation.")

        # --- Start of existing Qwen HF generation logic ---
        # (Keep the corrected HF generate logic from previous steps here)
        # ... includes apply_chat_template, processor call, move_inputs_to_device, model.generate, decode ...
        try:
            prompt = self._build_qwen_prompt(conversation)
        except Exception as e:
             raise RuntimeError(f"Failed to build Qwen prompt: {e}")

        try:
            inputs = self.hf_processor(
                text=[prompt], images=image_inputs, return_tensors="pt", padding=True
            )
        except Exception as e:
            print(f"Error processing inputs with Qwen HF processor: {e}")
            raise

        inputs = self.move_inputs_to_device(inputs)

        try:
            generate_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        except Exception as e:
             print(f"Error during Qwen HF model.generate: {e}")
             raise

        input_token_len = inputs['input_ids'].shape[1]
        generated_ids_only = generate_ids[:, input_token_len:]
        output_text = self.hf_processor.batch_decode(
            generated_ids_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text.strip()
        # --- End of existing Qwen HF generation logic ---


    def _generate_vllm(self, conversation: list, image_inputs: list[Image.Image], max_new_tokens: int) -> str:
        """ Generate using vLLM backend for Qwen. """
        if not self.model:
            raise RuntimeError("vLLM engine not loaded.")
        if not VLLM_AVAILABLE:
             raise RuntimeError("vLLM library not available.")

        # 1. Build the text prompt string
        prompt_str = self._build_qwen_prompt(conversation)
        # Ensure prompt includes correct Qwen placeholders (e.g., <img>, Picture 1:...)
        # This should be handled by apply_chat_template with the Qwen processor.

        # Optional: Use qwen-vl-utils for image preprocessing if available
        final_image_inputs = image_inputs
        if QWEN_UTILS_AVAILABLE and process_vision_info:
            try:
                # Note: process_vision_info might expect the message format
                # used in the example. We may need to adapt our 'conversation'
                # or mimic that structure if necessary. For now, let's assume
                # direct PIL list works, like LLaVA. If errors occur, revisit this.
                print("Note: Qwen-utils image processing step skipped for now. Using raw PIL images.")
                # Example structure if needed:
                # messages_for_utils = [{"role": "user", "content": [{"type": "image", "image": pil_img} for pil_img in image_inputs] + [{"type": "text", "text": "dummy text"}]}]
                # processed_data, _ = process_vision_info(messages_for_utils)
                # final_image_inputs = processed_data['image'] # Check actual output format
            except Exception as e:
                print(f"Warning: Failed to use qwen_vl_utils.process_vision_info: {e}. Using raw PIL images.")

        # 2. Define Sampling Parameters
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0,
             # Qwen might have specific stop tokens, check model card/tokenizer
             # stop_token_ids=[...] # Add specific stop token IDs if needed
        )

        # 3. Generate using vLLM engine
        try:
            request_dict = {
                "prompt": prompt_str,
                "multi_modal_data": {
                    "image": final_image_inputs # Pass PIL list
                }
            }

            outputs = self.model.generate(request_dict, sampling_params=sampling_params)

            if not outputs or outputs[0].outputs is None:
                 print("Error: Qwen vLLM generation returned empty output.")
                 return "[Qwen vLLM generation failed]"
            if outputs[0].outputs[0].finish_reason == 'length':
                 print("Warning: Max tokens reached during vLLM generation.")

            # 4. Extract output text
            generated_text = outputs[0].outputs[0].text
            return generated_text.strip()

        except Exception as e:
            print(f"Error during Qwen vLLM generation: {e}")
            print(f"Prompt: {prompt_str}")
            print(f"Number of images passed: {len(final_image_inputs)}")
            raise
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

    def _build_qwen_prompt(self, conversation: list) -> str:
        """ Applies chat template using HF processor. """
        if not self.hf_processor:
             # Fallback if processor failed: try simple concatenation (likely wrong format)
             print("Warning: HF Processor not available for Qwen prompt building.")
             return "\n".join([item['text'] for item in conversation[0]['content'] if item['type'] == 'text'])

        try:
            # Apply chat template to get the final prompt string including image placeholders
            prompt = self.hf_processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            # Debug: print(f"Qwen Prompt String for backend '{self.inference_backend}':\n{prompt}")
            return prompt
        except Exception as e:
             print(f"Error applying Qwen chat template: {e}")
             raise

