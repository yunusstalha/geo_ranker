# models/llava_vlm.py

from PIL import Image
import torch
import torch.nn.functional as F # Added for softmax
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
from typing import List, Dict, Union # Added typing

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
            self._load_vllm_engine()
        else:
            raise ValueError(f"Unsupported inference_backend: {self.inference_backend}")

    def _load_hf_model(self):
        """ Loads the HuggingFace model (existing logic). """
        print(f"Loading LLaVA model for HF backend: {self.model_name}...")
        start_time = time.time()

        # --- Start of existing HF loading logic ---
        quantization_config = None
        if self.use_quantization and self.hf_device != "cpu": # Quantization only on CUDA
            print("Applying 4-bit BitsAndBytes quantization for HF...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if self.use_bf16 else torch.float16
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
                quantization_config = None

        # If quantizing, device_map must be 'auto' for BNB usually
        if self.use_quantization and quantization_config:
            effective_device_map = "auto"
            model_dtype_arg = None # dtype handled by quantization_config or implicitly
            print(f"Using HF quantization. Setting device_map='{effective_device_map}' and model_dtype_arg=None.")

        try:
            # Load HF Processor (needed for both backends potentially)
            self.hf_processor = AutoProcessor.from_pretrained(self.model_name)
            # Load HF Model
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map=effective_device_map,
                torch_dtype=model_dtype_arg,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True if effective_device_map != "cpu" else False # Required when device_map is used
            )
            self.model.eval()
            # Determine the actual device after loading
            if hasattr(self.model, 'device'):
                self.hf_device = str(self.model.device) # Update hf_device based on actual placement
                print(f"HF Model loaded. Device map: {getattr(self.model, 'hf_device_map', 'N/A')}. Effective device: {self.hf_device}")
            else:
                print(f"HF Model loaded. Device map used: {effective_device_map}. Cannot determine final device from model object.")

        except Exception as e:
            print(f"Error loading LLaVA HF model {self.model_name}: {e}")
            import traceback
            traceback.print_exc()
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
            vllm_quantization = "awq" # Example, check vLLM docs for LLaVA support
            print(f"Applying vLLM quantization: {vllm_quantization}")

        # Use self.max_images_per_prompt passed during init
        engine_args = EngineArgs(
            model=self.model_name,
            quantization=vllm_quantization,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True, # Often required for VLMs
            limit_mm_per_prompt={"image": self.max_images_per_prompt},
        )
        print(f"vLLM EngineArgs: {engine_args}")

        try:
            self.model = LLM(**asdict(engine_args))
            print("vLLM Engine loaded successfully.")
        except Exception as e:
            print(f"Error loading vLLM engine for {self.model_name}: {e}")
            raise

        end_time = time.time()
        print(f"vLLM Engine loading took {end_time - start_time:.2f} seconds.")

    def _build_llava_prompt(self, conversation: list) -> str:
        """ Builds the LLaVA prompt string. """
        prompt_text_parts = []
        if not conversation or conversation[0].get("role") != "user":
            print("Warning: LLaVA prompt builder expects conversation starting with user role.")
            # Handle potential structure errors more gracefully
            # Attempt to find user content anyway or return a default error prompt
            user_content = []
            for msg in conversation:
                if msg.get("role") == "user":
                    user_content = msg.get("content", [])
                    break
            if not user_content:
                 return "ASSISTANT:" # Fallback if no user content found

        else:
             user_content = conversation[0].get("content", [])


        # Count actual images vs placeholders in conversation structure
        image_placeholders_in_conv = sum(1 for item in user_content if item.get("type") == "image")


        image_idx = 0
        for item in user_content:
            item_type = item.get("type")
            if item_type == "text":
                prompt_text_parts.append(item.get("text", ""))
            elif item_type == "image":
                # LLaVA typically uses one <image> token per actual image input passed to processor
                prompt_text_parts.append("<image>") # Use the placeholder LLaVA expects
                image_idx += 1
            else:
                print(f"Warning: Unknown item type in LLaVA conversation: {item_type}")

        full_text = "\n".join(prompt_text_parts)

        # Construct the standard USER/ASSISTANT prompt format
        # Handle potential missing system prompt if needed by model version
        # LLaVA 1.5 typically uses "USER: ...\nASSISTANT:"
        prompt = f"USER: {full_text}\nASSISTANT:"

        # Debugging log (optional)
        # print(f"Built LLaVA Prompt:\n{prompt}")
        # print(f"Number of image placeholders in prompt string: {prompt.count('<image>')}")

        return prompt


    # generate, _generate_hf, _generate_vllm remain the same
    # ... (paste existing methods here) ...
    def generate(self, conversation: Union[List, Dict], image_inputs: List[Image.Image], max_new_tokens: int = 256) -> str:
        """ Generate text using the selected backend. """
        if not isinstance(conversation, list):
             # Handle dict format if necessary, or raise error if only list is expected
             raise TypeError("LLaVA conversation must be a list.")

        if self.inference_backend == 'hf':
            return self._generate_hf(conversation, image_inputs, max_new_tokens)
        elif self.inference_backend == 'vllm':
            return self._generate_vllm(conversation, image_inputs, max_new_tokens)
        else:
            raise ValueError(f"Unsupported inference_backend: {self.inference_backend}")

    def _generate_hf(self, conversation: list, image_inputs: list[Image.Image], max_new_tokens: int) -> str:
        """ Generate using HuggingFace backend. """
        if not self.model or not self.hf_processor:
            raise RuntimeError("HF Model or processor not loaded for HF generation.")

        prompt_str = self._build_llava_prompt(conversation)
        try:
            inputs = self.hf_processor(
                text=prompt_str, images=image_inputs, return_tensors="pt", padding=True
            )
        except Exception as e:
            print(f"Error processing inputs with HF processor: {e}")
            raise

        inputs = self.move_inputs_to_device(inputs) # Use the corrected base class method

        # Generate with HF model
        # Use do_sample=False for deterministic scoring/preference unless exploring
        try:
             with torch.no_grad(): # Ensure no gradients are calculated
                 generate_ids = self.model.generate(
                     **inputs,
                     max_new_tokens=max_new_tokens,
                     do_sample=False,
                     pad_token_id=self.hf_processor.tokenizer.pad_token_id # Explicitly set pad token id
                 )
        except Exception as e:
             print(f"Error during LLaVA HF model.generate: {e}")
             # You might want to log more details about inputs here
             print(f"Input IDs shape: {inputs.get('input_ids').shape if inputs.get('input_ids') is not None else 'N/A'}")
             raise

        # Decode
        input_token_len = inputs['input_ids'].shape[1]
        # Handle potential case where generate_ids is shorter than input_ids (shouldn't happen with max_new_tokens > 0)
        if generate_ids.shape[1] <= input_token_len:
             print("Warning: Generation produced no new tokens or fewer tokens than input.")
             generated_ids_only = torch.tensor([[]], dtype=torch.long, device=generate_ids.device) # Empty tensor
        else:
             generated_ids_only = generate_ids[:, input_token_len:]

        output_text = self.hf_processor.batch_decode(
            generated_ids_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    def _generate_vllm(self, conversation: list, image_inputs: list[Image.Image], max_new_tokens: int) -> str:
        """ Generate using vLLM backend. """
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

            if not outputs or not outputs[0].outputs:
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
            # Re-raise the exception to make it clear generation failed
            raise RuntimeError(f"LLaVA vLLM generation failed: {e}") from e


    @torch.no_grad() # Disable gradient calculation for inference
    def score_multiple_choice(self, conversation: Union[List, Dict], image_inputs: List[Image.Image], choices: List[str]) -> Dict[str, float]:
        """ Scores choices using HF backend logits. """
        if self.inference_backend != 'hf':
            raise NotImplementedError("score_multiple_choice is only implemented for the 'hf' backend.")
        if not self.model or not self.hf_processor:
            raise RuntimeError("HF Model or processor not loaded for score_multiple_choice.")
        if not self.hf_processor.tokenizer:
             raise RuntimeError("HF Processor does not have a tokenizer needed for scoring.")

        # Ensure conversation is a list for LLaVA
        if not isinstance(conversation, list):
            raise TypeError("LLaVA conversation must be a list for score_multiple_choice.")

        prompt_str = self._build_llava_prompt(conversation)
        tokenizer = self.hf_processor.tokenizer
        space_token_id = tokenizer.encode(" ", add_special_tokens=False)

        space_token_id = 29871

        # --- Tokenize Choices ---
        choice_token_ids = []
        choice_token_map = {} # Map first token ID back to original choice string
        # print(f"Tokenizing choices for LLaVA: {choices}")
        for choice in choices:
            choice_clean = choice.strip()
            token_ids = tokenizer.encode(choice_clean, add_special_tokens=False)

            target_token_id = -1 # Initialize

            if not token_ids:
                print(f"Warning: Choice '{choice}' resulted in empty token list. Skipping.")
                continue

            # *** LIKERT FIX LOGIC ***
            if len(token_ids) == 2 and token_ids[0] == space_token_id:
                # If encoded as [space, actual_token], use the second one
                target_token_id = token_ids[1]
                # Optional: Log this case
                # print(f"  Info: Choice '{choice_clean}' tokenized as [space, word/digit]. Using second token ID {target_token_id}.")
            elif len(token_ids) == 1:
                # If encoded as a single token, use that one
                target_token_id = token_ids[0]
                # Optional: Log this case
                # print(f"  Info: Choice '{choice_clean}' tokenized as single token ID {target_token_id}.")
            else:
                # Handle unexpected cases (e.g., >2 tokens, or multiple tokens not starting with space)
                target_token_id = token_ids[1] # Default to first token as fallback
                decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids)
                print(f"  Warning: Choice '{choice_clean}' tokenized unexpectedly into {len(token_ids)} tokens: {decoded_tokens}. Using first token ID {target_token_id} ('{tokenizer.decode([target_token_id])}') as fallback.")

            if target_token_id != -1:
                 choice_token_ids.append(target_token_id)
                 # Use the *original* choice string as the key in the map
                 choice_token_map[target_token_id] = choice
                 # Debug: print(f"  Mapping target token {target_token_id} to original choice '{choice}'")


        # print(f"Target token IDs: {choice_token_ids}") # Should now contain digit tokens etc.
        if not choice_token_ids:
             print("Error: No valid token IDs found for any choices.")
             return {choice: 0.0 for choice in choices} # Return zero probabilities


        # --- Process Prompt and Images ---
        try:
            inputs = self.hf_processor(
                text=prompt_str, images=image_inputs, return_tensors="pt", padding=True
            )
        except Exception as e:
            print(f"Error processing inputs for scoring: {e}")
            raise

        inputs = self.move_inputs_to_device(inputs) # Use the base class method

        # --- Get Logits for the NEXT token ---
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            next_token_logits = outputs.scores[0][0] # Shape: [vocab_size]

        except Exception as e:
             print(f"Error during model generation/logit retrieval: {e}")
             raise RuntimeError("Failed to obtain logits for scoring.") from e

        # --- Calculate Probabilities for Choices ---
        # Extract logits only for the target token IDs
        # Ensure target IDs are valid indices for the logits tensor
        valid_choice_token_ids = [tid for tid in choice_token_ids if tid < next_token_logits.shape[0]]
        if len(valid_choice_token_ids) != len(choice_token_ids):
            print("Warning: Some choice token IDs were out of vocabulary range.")
            # Update choice_token_map and choice_token_ids if needed, or handle below

        choice_logits = next_token_logits[valid_choice_token_ids]

        # Apply Softmax to get probabilities *over the valid choices only*
        choice_probs = F.softmax(choice_logits, dim=-1)

        # Create the result dictionary mapping choice string to probability
        result_probs = {}
        valid_choices_found = set()
        # Use valid_choice_token_ids which contains only IDs within vocab range
        valid_choice_token_ids_in_vocab = [tid for tid in choice_token_ids if tid < next_token_logits.shape[0]]
        if len(valid_choice_token_ids_in_vocab) != len(choice_token_ids):
             print(f"Warning: Some choice token IDs were out of vocabulary range ({next_token_logits.shape[0]}).")
             # Filter choice_token_ids and rebuild map if necessary, but logic below handles missing keys

        if not valid_choice_token_ids_in_vocab:
             print("Error: None of the target token IDs are valid for the model's vocabulary.")
             return {choice: 0.0 for choice in choices}

        choice_logits = next_token_logits[valid_choice_token_ids_in_vocab]
        choice_probs = F.softmax(choice_logits, dim=-1)


        for i, token_id in enumerate(valid_choice_token_ids_in_vocab):
            # Map the *valid* token ID back to the *original* choice string
            original_choice = choice_token_map.get(token_id)
            if original_choice:
                 # Assign the probability corresponding to this valid token
                 result_probs[original_choice] = choice_probs[i].item()
                 valid_choices_found.add(original_choice)
                 # Debug: print(f"  Assigned prob {choice_probs[i].item():.4f} to choice '{original_choice}' (token {token_id})")

            else:
                 # This indicates an internal logic error in choice_token_map
                 print(f"Internal Error: Could not map valid token ID {token_id} back to an original choice.")

        # Ensure all original choices are present in the final dict, assigning 0.0 if they were invalid/skipped
        final_result = {choice: result_probs.get(choice, 0.0) for choice in choices}


        # print(f"Calculated LLaVA probabilities: {final_result}")
        return final_result


    # move_inputs_to_device is now in BaseVLM