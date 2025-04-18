# models/qwen_vlm.py

import os
from PIL import Image
import torch
import torch.nn.functional as F # Added for softmax

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from dataclasses import asdict
try:
    from vllm import LLM, EngineArgs, SamplingParams
    VLLM_AVAILABLE = True
    # from qwen_vl_utils import process_vision_info # Comment out if not used or causing issues
    # QWEN_UTILS_AVAILABLE = True
    QWEN_UTILS_AVAILABLE = False # Assume not used for now
except ImportError:
    process_vision_info = None
    QWEN_UTILS_AVAILABLE = False
    LLM = None
    SamplingParams = None
    VLLM_AVAILABLE = False
    print("Warning: vLLM or qwen_vl_utils not found. vLLM backend or specific Qwen utils will not be available.")


from .base_vlm import BaseVLM
import time
from typing import List, Dict, Union # Added typing

import traceback # Import traceback for detailed error logging

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
        # Pass potentially resolved device to super()
        resolved_device = device
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(model_name=model_name, device=resolved_device, inference_backend=inference_backend)

    def _load_model(self):
        """ Load model based on the selected inference backend. """
        if self.inference_backend == 'hf':
            self._load_hf_model()
        elif self.inference_backend == 'vllm':
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM backend selected, but vLLM library is not installed or failed to import.")
            self._load_vllm_engine() # This might fail if Qwen-VL not supported well by vLLM
        else:
            raise ValueError(f"Unsupported inference_backend: {self.inference_backend}")

    def _load_hf_model(self):
        """ Loads the HuggingFace Qwen model (existing logic). """
        print(f"Loading Qwen model for HF backend: {self.model_name}...")
        start_time = time.time()
        quantization_config_bnb = None
        if self.use_quantization and self.hf_device != "cpu": # Quantization only on CUDA
            print("Applying 4-bit BitsAndBytes quantization for HF...")
            quantization_config_bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.use_bf16 else torch.float16,
                 # Qwen might need specific bnb settings, check model card if issues arise
            )

        dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        effective_device_map = self.hf_device
        model_dtype_arg = dtype

        if self.hf_device == "cpu":
            model_dtype_arg = torch.float32
            effective_device_map = "cpu"
            print("Warning: Running HF on CPU, forcing float32.")
            if self.use_quantization:
                print("Warning: Quantization requested but running on CPU. Disabling HF quantization.")
                self.use_quantization = False # Ensure flag is off
                quantization_config_bnb = None

        # If quantizing, device_map must be 'auto' for BNB usually
        if self.use_quantization and quantization_config_bnb:
            effective_device_map = "auto"
            model_dtype_arg = None # dtype handled by quantization_config or implicitly
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
             # Determine the actual device after loading
            if hasattr(self.model, 'device'):
                self.hf_device = str(self.model.device) # Update hf_device based on actual placement
                print(f"HF Model loaded. Device map: {getattr(self.model, 'hf_device_map', 'N/A')}. Effective device: {self.hf_device}")
            else:
                 # Qwen model might store device differently or not have a top-level .device
                 # We rely on the device_map argument primarily here.
                 print(f"HF Model loaded. Device map used: {effective_device_map}. Cannot determine final device from model object.")


        except Exception as e:
            print(f"Error loading Qwen HF model {self.model_name}: {e}")
            import traceback
            traceback.print_exc()
            raise
        end_time = time.time()
        print(f"HF Model loading took {end_time - start_time:.2f} seconds.")


    def _load_vllm_engine(self):
        """ Loads the vLLM engine for Qwen. """
        print(f"Loading Qwen model for vLLM backend: {self.model_name}...")
        start_time = time.time()

        print("Loading HF Processor for text processing (needed for prompt building)...")
        try:
            self.hf_processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Could not load HF Processor {self.model_name}: {e}")
            self.hf_processor = None # Prompt building might fail

        vllm_quantization = None
        if self.use_quantization:
            vllm_quantization = "awq" # Check vLLM+Qwen docs for supported methods
            print(f"Applying vLLM quantization: {vllm_quantization}")

        # Use self.max_images_per_prompt passed during init
        engine_args = EngineArgs(
            model=self.model_name,
            quantization=vllm_quantization,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True, # Essential for Qwen
            limit_mm_per_prompt={"image": self.max_images_per_prompt},
            # Consider max_model_len if needed
        )
        print(f"vLLM EngineArgs: {engine_args}")

        try:
            self.model = LLM(**asdict(engine_args))
            print("vLLM Engine loaded successfully for Qwen.")
        except Exception as e:
            print(f"Error loading vLLM engine for Qwen model {self.model_name}: {e}")
            raise

        end_time = time.time()
        print(f"vLLM Engine loading took {end_time - start_time:.2f} seconds.")


    # generate, _generate_hf, _generate_vllm, _build_qwen_prompt remain the same
    # ... (paste existing methods here) ...
    def generate(self, conversation: Union[List, Dict], image_inputs: List[Image.Image], max_new_tokens: int = 256) -> str:
        """ Generate text using the selected backend. """
        # Qwen uses list internally, ensure input matches
        if not isinstance(conversation, list):
             raise TypeError("Qwen conversation must be a list.")

        if self.inference_backend == 'hf':
            return self._generate_hf(conversation, image_inputs, max_new_tokens)
        elif self.inference_backend == 'vllm':
            return self._generate_vllm(conversation, image_inputs, max_new_tokens)
        else:
            raise ValueError(f"Unsupported inference_backend: {self.inference_backend}")

    def _generate_hf(self, conversation: list, image_inputs: list[Image.Image], max_new_tokens: int) -> str:
        """ Generate using HuggingFace backend. """
        if not self.model or not self.hf_processor:
            raise RuntimeError("HF Model or processor not loaded for Qwen HF generation.")

        try:
            # Let the processor handle image tags and text assembly
            prompt_build_start_time = time.time()
            # Qwen processor needs conversation structure directly
            text = self.hf_processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.hf_processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
            prompt_build_end_time = time.time()
            #print(f"Qwen HF Prompt build/process time: {prompt_build_end_time - prompt_build_start_time:.2f}s")

        except Exception as e:
            print(f"Error during Qwen HF processing/prompt building: {e}")
            print(f"Conversation structure: {conversation}") # Log conversation on error
            raise

        inputs = self.move_inputs_to_device(inputs) # Use corrected base class method

        try:
            gen_start_time = time.time()
            with torch.no_grad():
                 generate_ids = self.model.generate(
                        **inputs,
                     max_new_tokens=max_new_tokens,
                     do_sample=False,
                     pad_token_id=self.hf_processor.tokenizer.eos_token_id # Qwen often uses eos_token_id for padding
                 )
            gen_end_time = time.time()
            #print(f"Qwen HF Generation time: {gen_end_time - gen_start_time:.2f}s")

        except Exception as e:
            print(f"Error during Qwen HF model.generate: {e}")
            print(f"Input IDs shape: {inputs.get('input_ids').shape if inputs.get('input_ids') is not None else 'N/A'}")
            print(f"Pixel Values shape: {inputs.get('pixel_values').shape if inputs.get('pixel_values') is not None else 'N/A'}")
            print(f"Attention Mask shape: {inputs.get('attention_mask').shape if inputs.get('attention_mask') is not None else 'N/A'}")
            raise

        # Decode
        decode_start_time = time.time()
        input_token_len = inputs['input_ids'].shape[1]

        if generate_ids.shape[1] <= input_token_len:
             print("Warning: Generation produced no new tokens or fewer tokens than input.")
             generated_ids_only = torch.tensor([[]], dtype=torch.long, device=generate_ids.device) # Empty tensor
        else:
             generated_ids_only = generate_ids[:, input_token_len:]

        # Use tokenizer directly for decoding Qwen outputs
        output_text = self.hf_processor.tokenizer.batch_decode(
            generated_ids_only, skip_special_tokens=True # Skip special tokens like <|im_end|>
        )[0]
        decode_end_time = time.time()
        #print(f"Qwen HF Decode time: {decode_end_time - decode_start_time:.2f}s")

        return output_text.strip()


    def _generate_vllm(self, conversation: list, image_inputs: list[Image.Image], max_new_tokens: int) -> str:
        """ Generate using vLLM backend for Qwen. """
        if not self.model:
            raise RuntimeError("vLLM engine not loaded.")
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM library not available.")
        if not self.hf_processor:
            raise RuntimeError("Qwen HF Processor required for vLLM prompt building but not loaded.")


        # 1. Build the text prompt string using the HF processor's template
        try:
            prompt_str = self.hf_processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
             print(f"Error applying Qwen chat template for vLLM: {e}")
             raise RuntimeError(f"Failed to build Qwen prompt for vLLM: {e}") from e

        # Image handling: vLLM typically takes PIL images directly
        final_image_inputs = image_inputs
        # Qwen-utils processing step (currently disabled/optional)
        # if QWEN_UTILS_AVAILABLE and process_vision_info: ...

        # 2. Define Sampling Parameters
        # Check Qwen tokenizer for appropriate stop tokens (e.g., <|im_end|>, <|endoftext|>)
        stop_token_ids = [self.hf_processor.tokenizer.eos_token_id]
        # Add other relevant stop tokens if necessary
        # stop_token_ids.append(self.hf_processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0,
            stop_token_ids=stop_token_ids
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

            if not outputs or not outputs[0].outputs:
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
            raise RuntimeError(f"Qwen vLLM generation failed: {e}") from e

    def _build_qwen_prompt(self, conversation: list) -> str:
         """ Applies chat template using HF processor. Required by _generate_hf. """
         if not self.hf_processor:
             # Fallback if processor failed: try simple concatenation (likely wrong format)
             print("Warning: HF Processor not available for Qwen prompt building.")
             # Basic fallback assuming single turn user message
             text_parts = []
             if conversation and conversation[0].get("role") == "user":
                  content = conversation[0].get("content", [])
                  for item in content:
                       if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                       elif item.get("type") == "image":
                            text_parts.append("Picture X:") # Generic placeholder
             return "\n".join(text_parts) + "\nAssistant:" # Minimal fallback prompt


         try:
             # Apply chat template to get the final prompt string including image placeholders
             prompt = self.hf_processor.apply_chat_template(
                 conversation,
                 tokenize=False,
                 add_generation_prompt=True # Adds the prompt for the assistant's turn
             )
             # Debug: print(f"Qwen Prompt String for backend '{self.inference_backend}':\n{prompt}")
             return prompt
         except Exception as e:
             print(f"Error applying Qwen chat template: {e}")
             print(f"Conversation structure: {conversation}")
             raise RuntimeError(f"Failed to build Qwen prompt: {e}") from e


    @torch.no_grad()
    def score_multiple_choice(self, conversation: Union[List, Dict], image_inputs: List[Image.Image], choices: List[str]) -> Dict[str, float]:
        """
        Scores choices using HF backend logits for Qwen.
        Uses generate with **inputs syntax and max_new_tokens > 1.
        """
        if self.inference_backend != 'hf':
            raise NotImplementedError("score_multiple_choice is only implemented for the 'hf' backend.")
        # ... (other checks: model, processor, tokenizer, conversation type) ...
        if not self.model or not self.hf_processor: raise RuntimeError("HF Model/processor not loaded")
        if not self.hf_processor.tokenizer: raise RuntimeError("HF Processor tokenizer missing")
        if not isinstance(conversation, list): raise TypeError("Qwen conversation must be a list")

        tokenizer = self.hf_processor.tokenizer

        # --- Determine Space Token ID (remains the same) ---
        space_token_id = tokenizer.encode(" ", add_special_tokens=False)
        if len(space_token_id) == 1: space_token_id = space_token_id[0]
        else: space_token_id = -1 # Ambiguous

        # --- Tokenize Choices (remains the same) ---
        choice_token_ids = []
        choice_token_map = {}
        # print(f"Tokenizing choices for Qwen: {choices}")
        for choice in choices:
             choice_clean = choice.strip()
             token_ids = tokenizer.encode(choice_clean, add_special_tokens=False)
             target_token_id = -1
             if not token_ids: continue
             if space_token_id != -1 and len(token_ids) == 2 and token_ids[0] == space_token_id:
                 target_token_id = token_ids[1]
             elif len(token_ids) == 1:
                 target_token_id = token_ids[0]
             else:
                 target_token_id = token_ids[0] # Fallback
             if target_token_id != -1:
                  choice_token_ids.append(target_token_id)
                  choice_token_map[target_token_id] = choice
        # print(f"Target Qwen token IDs: {choice_token_ids}")
        if not choice_token_ids:
              print("Error: No valid token IDs found for any choices.")
              return {choice: 0.0 for choice in choices}

        # --- Process Prompt and Images (remains the same) ---
        try:
            text = self.hf_processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            inputs = self.hf_processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
        except Exception as e:
            print(f"Error processing inputs/prompt for Qwen scoring: {e}")
            raise
        inputs = self.move_inputs_to_device(inputs)

        # --- Get Logits for the NEXT token (Using generate with **inputs) ---
        next_token_logits = None
        # Define how many tokens to generate (needs to be > 1 potentially)
        try:
            # *** CORRECTION HERE: Use **inputs unpacking ***
            outputs = self.model.generate(
                **inputs,                           # Pass all processed inputs
                max_new_tokens=1,
                output_scores=True,                 # Still request scores
                return_dict_in_generate=True,
                do_sample=False,
                num_beams=1,                     # Force greedy explicitly
                pad_token_id=tokenizer.eos_token_id
            )

            # Check if scores are available and valid for the first token
            if hasattr(outputs, 'scores') and outputs.scores is not None and len(outputs.scores) > 0:
                if outputs.scores[0] is not None:
                    next_token_logits = outputs.scores[0][0] # Batch index 0, First token's scores
                    # print("Logits successfully retrieved via model.generate (first token scores).")
                else:
                    print("Warning: model.generate returned scores tuple, but the first element (scores for first token) is None.")
            else:
                print("Warning: model.generate did not return valid scores attribute or it was empty/None.")
                # Log details if scores are missing
                # print(f"Generate output keys: {hasattr(outputs, 'keys') and outputs.keys()}")
                # print(f"Generate output scores attribute: {getattr(outputs, 'scores', 'MISSING')}")


        except Exception as e:
             print(f"Error during Qwen model generation/logit retrieval using **inputs: {e}")
             traceback.print_exc()
             raise RuntimeError("Failed to obtain logits for Qwen scoring using generate with **inputs.") from e

        # --- Ensure logits were obtained ---
        if next_token_logits is None:
             raise RuntimeError("Logit retrieval failed (next_token_logits is None after generate). Check warnings above.")

        # --- Calculate Probabilities for Choices ---
        # (Rest of the logic remains the same: filter valid tokens, softmax, map to choices)
        # ...
        result_probs = {}
        valid_choice_token_ids_in_vocab = [tid for tid in choice_token_ids if tid >= 0 and tid < next_token_logits.shape[0]]
        if len(valid_choice_token_ids_in_vocab) != len(choice_token_ids):
             invalid_ids = set(choice_token_ids) - set(valid_choice_token_ids_in_vocab)
             print(f"Warning: Some choice token IDs {invalid_ids} were out of vocabulary range ({next_token_logits.shape[0]}).")
        if not valid_choice_token_ids_in_vocab:
             print("Error: None of the target token IDs are valid for the model's vocabulary.")
             return {choice: 0.0 for choice in choices}

        choice_logits = next_token_logits[valid_choice_token_ids_in_vocab]
        choice_probs = F.softmax(choice_logits, dim=-1)

        for i, token_id in enumerate(valid_choice_token_ids_in_vocab):
            original_choice = choice_token_map.get(token_id)
            if original_choice:
                 result_probs[original_choice] = choice_probs[i].item()
        final_result = {choice: result_probs.get(choice, 0.0) for choice in choices}

        # print(f"Calculated Qwen probabilities: {final_result}")
        return final_result



    # move_inputs_to_device is now in BaseVLM