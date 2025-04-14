# models/llava_vlm.py

from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration, # Specific Llava model class
    BitsAndBytesConfig
)
from .base_vlm import BaseVLM
import time

class LlavaVLM(BaseVLM):
    def __init__(self,
                 model_name: str = "llava-hf/llava-1.5-7b-hf", # Example LLaVA model
                 device: str = "auto",
                 use_bf16: bool = True,
                 use_quantization: bool = False # Control quantization
                ):
        """
        Initialize the LlavaVLM model and processor.
        """
        self.use_bf16 = use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.use_quantization = use_quantization
        super().__init__(model_name=model_name, device=device)

    def _load_model(self):
        """
        Load the LLaVA model and processor.
        """
        print(f"Loading LLaVA model: {self.model_name}...")
        start_time = time.time()

        quantization_config = None
        if self.use_quantization:
            print("Applying 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", # nf4 is common for LLaVA
                bnb_4bit_compute_dtype=torch.bfloat16 if self.use_bf16 else torch.float16
            )

        dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        effective_device_map = self.device
        model_dtype_arg = dtype

        # Handle CPU case explicitly
        if self.device == "cpu":
            model_dtype_arg = torch.float32
            effective_device_map = "cpu" # Explicitly set device_map for CPU
            print("Warning: Running on CPU, forcing float32. Performance will be slow.")
            if self.use_quantization:
                print("Warning: Quantization requested but running on CPU. Disabling quantization.")
                self.use_quantization = False
                quantization_config = None

        # Adjust parameters if quantization is enabled
        if self.use_quantization:
             effective_device_map = "auto"
             model_dtype_arg = None # Let BnB handle dtype
             print(f"Using quantization. Setting device_map='{effective_device_map}' and model_dtype_arg=None.")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map=effective_device_map,
                torch_dtype=model_dtype_arg,
                quantization_config=quantization_config,
                # *** FIX: Set low_cpu_mem_usage=True whenever device_map is used ***
                # (It's generally safe and required when device_map is not None/implicitly CPU)
                low_cpu_mem_usage=True
            )
            self.model.eval()

            # This manual move is likely redundant if device_map is used,
            # but harmless if model is already on the target device.
            # Useful primarily if device_map was None (e.g., explicit CPU loading failed earlier)
            # or if fine-tuning requires manual placement.
            if self.device != "auto" and not getattr(self.model, 'hf_device_map', None):
                 print(f"Manually moving model to {self.device} as device_map info is not available.")
                 self.model.to(self.device)

            print(f"Model loaded. Device map: {getattr(self.model, 'hf_device_map', 'N/A')}. Effective device: {self.model.device}")


        except Exception as e:
            print(f"Error loading LLaVA model {self.model_name}: {e}")
            raise

        end_time = time.time()
        print(f"Model loading took {end_time - start_time:.2f} seconds.")
    def _build_llava_prompt(self, conversation: list) -> str:
        """
        Constructs the prompt string specifically for LLaVA 1.5 format
        from a conversation list structure.
        """
        prompt_text_parts = []
        image_placeholder_count = 0

        if conversation and conversation[0]["role"] == "user":
            user_content = conversation[0]["content"]
            for item in user_content:
                if item["type"] == "text":
                    prompt_text_parts.append(item["text"])
                elif item["type"] == "image":
                    # Add the standard LLaVA placeholder where image conceptually belongs
                    # LLaVA 1.5 usually expects <image> marker(s)
                    prompt_text_parts.append("<image>")
                    image_placeholder_count += 1
        else:
            print("Warning: LLaVA prompt builder expects conversation starting with user role.")
            return "ASSISTANT:" # Fallback

        # Combine text parts and placeholders
        full_text = "\n".join(prompt_text_parts)

        # Construct the final prompt string
        # Ensure this matches the specific LLaVA model's expected format
        prompt = f"USER: {full_text}\nASSISTANT:"
        # print(f"LLaVA Prompt String:\n{prompt}") # For debugging
        return prompt


    def generate(self, conversation: list, image_inputs: list[Image.Image], max_new_tokens: int = 256) -> str:
        """
        Generate a response using the LLaVA model.
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model and processor not loaded.")

        # 1. Construct the prompt string in LLaVA format
        prompt = self._build_llava_prompt(conversation)

        # 2. Process images and text prompt using LLaVA processor
        try:
            inputs = self.processor(
                text=prompt,          # Pass the formatted prompt string
                images=image_inputs,  # Pass PIL images
                return_tensors="pt",
                padding=True
            )
        except Exception as e:
            print(f"Error processing inputs with LLaVA processor: {e}")
            print(f"Prompt: {prompt}")
            print(f"Number of images: {len(image_inputs)}")
            raise

        # 3. Move inputs to device
        inputs = self.move_inputs_to_device(inputs)

        # 4. Generate
        generate_ids = self.model.generate(
            **inputs, # Pass all processed inputs
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

        # 5. Decode, skipping prompt tokens AND special tokens
        input_token_len = inputs['input_ids'].shape[1]
        generated_ids_only = generate_ids[:, input_token_len:]

        # Use clean_up_tokenization_spaces=False as recommended for LLaVA
        output_text = self.processor.batch_decode(
            generated_ids_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    def move_inputs_to_device(self, inputs):
        """
        Move inputs to the correct device.
        """
        if self.device == "auto":
            return inputs
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        return inputs