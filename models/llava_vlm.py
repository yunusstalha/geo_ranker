# models/llava_vlm.py

from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig
)
# Try importing vLLM components, handle error if not installed
from dataclasses import asdict
try:
    from vllm import LLM, EngineArgs, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    SamplingParams = None
    VLLM_AVAILABLE = False
    print("Warning: vLLM not found. vLLM backend will not be available.")

from .base_vlm import BaseVLM
import time
import base64 # Ensure imported here too
import io     # Ensure imported here too

class LlavaVLM(BaseVLM):
    # Update __init__ signature to accept backend args
    def __init__(self,
                 model_name: str = "llava-hf/llava-1.5-7b-hf",
                 device: str = "auto", # For HF
                 use_bf16: bool = True, # For HF
                 use_quantization: bool = False, # Controls quantization for both
                 inference_backend: str = 'hf', # 'hf' or 'vllm'
                 tensor_parallel_size: int = 1, # For vLLM
                 max_images_per_prompt: int = 5, # For vLLM
                ):
        """ Initialize LLaVA for HF or vLLM backend. """
        self.use_bf16 = use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        # Store backend-specific args before calling super
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
            self._load_vllm_engine()
        else:
            raise ValueError(f"Unsupported inference_backend: {self.inference_backend}")

    def _load_hf_model(self):
        """ Loads the HuggingFace model (existing logic). """
        print(f"Loading LLaVA model for HF backend: {self.model_name}...")
        start_time = time.time()

        # --- Start of existing HF loading logic ---
        quantization_config = None
        if self.use_quantization:
            print("Applying 4-bit BitsAndBytes quantization for HF...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if self.use_bf16 else torch.float16
            )

        dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        effective_device_map = self.hf_device # Use hf_device stored in base class
        model_dtype_arg = dtype

        if self.hf_device == "cpu":
            model_dtype_arg = torch.float32
            effective_device_map = "cpu"
            print("Warning: Running HF on CPU, forcing float32.")
            if self.use_quantization:
                print("Warning: Quantization requested but running on CPU. Disabling HF quantization.")
                self.use_quantization = False
                quantization_config = None

        if self.use_quantization:
             effective_device_map = "auto"
             model_dtype_arg = None
             print(f"Using HF quantization. Setting device_map='{effective_device_map}' and model_dtype_arg=None.")

        try:
            # Load HF Processor (might be needed for vLLM text prep too)
            self.hf_processor = AutoProcessor.from_pretrained(self.model_name)
            # Load HF Model
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map=effective_device_map,
                torch_dtype=model_dtype_arg,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True # Required when device_map is used
            )
            self.model.eval()
            print(f"HF Model loaded. Device map: {getattr(self.model, 'hf_device_map', 'N/A')}. Effective device: {self.model.device}")
        except Exception as e:
            print(f"Error loading LLaVA HF model {self.model_name}: {e}")
            raise
        # --- End of existing HF loading logic ---
        end_time = time.time()
        print(f"HF Model loading took {end_time - start_time:.2f} seconds.")

    def _load_vllm_engine(self):
        """ Loads the vLLM engine for LLaVA. """
        print(f"Loading LLaVA model for vLLM backend: {self.model_name}...")
        start_time = time.time()

        print("Loading HF Processor for text processing...")
        try:
             self.hf_processor = AutoProcessor.from_pretrained(self.model_name)
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
            print("vLLM Engine loaded successfully.")
        except Exception as e:
            print(f"Error loading vLLM engine for {self.model_name}: {e}")
            raise

        end_time = time.time()
        print(f"vLLM Engine loading took {end_time - start_time:.2f} seconds.")


    def _build_llava_prompt(self, conversation: list) -> str:
        """ Builds the LLaVA prompt string (remains the same logic). """
        # (Keep the existing _build_llava_prompt method here)
        prompt_text_parts = []
        image_placeholder_count = 0 # Count conceptual image placements

        if conversation and conversation[0]["role"] == "user":
            user_content = conversation[0]["content"]
            for item in user_content:
                if item["type"] == "text":
                    prompt_text_parts.append(item["text"])
                elif item["type"] == "image":
                    # Use the special image token expected by LLaVA/vLLM multimodal
                    # Check processor/model config for the correct token (e.g., '<image>')
                    # Default LLaVA 1.5 token is often '<image>' handled by processor/vLLM.
                    # We add it conceptually here; final handling depends on model/vLLM version.
                    prompt_text_parts.append("<image>")
                    image_placeholder_count += 1
        else:
            print("Warning: LLaVA prompt builder expects conversation starting with user role.")
            return "ASSISTANT:" # Fallback

        full_text = "\n".join(prompt_text_parts)
        prompt = f"USER: {full_text}\nASSISTANT:"
        # Debug: print(f"LLaVA Prompt String for backend '{self.inference_backend}':\n{prompt}")
        return prompt


    def generate(self, conversation: list, image_inputs: list[Image.Image], max_new_tokens: int = 256) -> str:
        """ Generate text using the selected backend. """
        if self.inference_backend == 'hf':
            return self._generate_hf(conversation, image_inputs, max_new_tokens)
        elif self.inference_backend == 'vllm':
            return self._generate_vllm(conversation, image_inputs, max_new_tokens)
        else:
            raise ValueError(f"Unsupported inference_backend: {self.inference_backend}")

    def _generate_hf(self, conversation: list, image_inputs: list[Image.Image], max_new_tokens: int) -> str:
        """ Generate using HuggingFace backend (existing logic). """
        if not self.model or not self.hf_processor:
            raise RuntimeError("HF Model or processor not loaded for HF generation.")

        # 1. Build prompt using the helper method
        prompt_str = self._build_llava_prompt(conversation)

        # 2. Process inputs using HF processor
        try:
            inputs = self.hf_processor(
                text=prompt_str, images=image_inputs, return_tensors="pt", padding=True
            )
        except Exception as e:
             print(f"Error processing inputs with HF processor: {e}")
             raise

        # 3. Move inputs to device
        inputs = self.move_inputs_to_device(inputs) # Use the helper

        # 4. Generate with HF model
        generate_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

        # 5. Decode
        input_token_len = inputs['input_ids'].shape[1]
        generated_ids_only = generate_ids[:, input_token_len:]
        output_text = self.hf_processor.batch_decode(
            generated_ids_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text.strip()

    def _generate_vllm(self, conversation: list, image_inputs: list[Image.Image], max_new_tokens: int) -> str:
        """ Generate using vLLM backend (Simplified based on example). """
        if not self.model:
            raise RuntimeError("vLLM engine not loaded.")
        if not VLLM_AVAILABLE:
             raise RuntimeError("vLLM library not available.")

        # 1. Build the text prompt string
        prompt_str = self._build_llava_prompt(conversation)
        # Ensure prompt_str contains placeholders like <image> correctly

        # 2. Define Sampling Parameters
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0, # For deterministic output
        )

        # 3. Generate using vLLM engine (passing PIL images directly)
        try:
            # Create the request dictionary matching the example format
            request_dict = {
                "prompt": prompt_str,
                "multi_modal_data": {
                    "image": image_inputs # Pass the list of PIL images directly
                }
            }

            outputs = self.model.generate(request_dict, sampling_params=sampling_params)

            if not outputs or outputs[0].outputs is None:
                 print("Error: LLaVA vLLM generation returned empty output.")
                 return "[LLaVA vLLM generation failed]"
            if outputs[0].outputs[0].finish_reason == 'length':
                 print("Warning: Max tokens reached during vLLM generation.")

            # 4. Extract output text
            generated_text = outputs[0].outputs[0].text
            return generated_text.strip()

        except Exception as e:
            print(f"Error during LLaVA vLLM generation: {e}")
            print(f"Prompt: {prompt_str}")
            print(f"Number of images passed: {len(image_inputs)}")
            raise
        except Exception as e:
            print(f"Error during vLLM generation: {e}")
            print(f"Prompt: {prompt_str}")
            print(f"Multi-modal data keys: {vllm_image_inputs.keys() if vllm_image_inputs else 'None'}")
            raise
    def move_inputs_to_device(self, inputs):
        """ Move inputs to the correct device (CPU/GPU) for HF. """
        if self.hf_device == "cuda":
            for key in inputs.keys():
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
        return inputs