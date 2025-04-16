import argparse
import torch
from PIL import Image
import json
import time
import os
import re # For parsing
import numpy as np # For likert expected value

# Import VLM classes and prompt builders
from models import QwenVLM, LlavaVLM
# Import the specific prompt builders or the getter function
from prompts.prompt_templates import get_prompt_builder

# --- Example Image Paths ---
DEFAULT_EXAMPLE_QUERY_IMG = "/research/nfs_yilmaz_15/yunus/data/VIGOR/SanFrancisco/panorama/ScGwgVhnS9mLGl47wRnicQ,37.752695,-122.437849,.jpg"
DEFAULT_EXAMPLE_SAT_IMG_2 = "/research/nfs_yilmaz_15/yunus/data/VIGOR/SanFrancisco/satellite/satellite_37.752731622631885_-122.43810708310743.png" # Incorrect match
DEFAULT_EXAMPLE_SAT_IMG_1 = "/research/nfs_yilmaz_15/yunus/data/VIGOR/SanFrancisco/satellite/satellite_37.752731622631885_-122.43769208587767.png" # Correct match

def check_image_path(path, label="Image"):
    """ Checks if path exists, prints warning if not. """
    if not path: # Handle None or empty string
        print(f"Warning: No path provided for {label}.")
        return None
    # Allow non-existent paths for testing, but print warning
    if not os.path.exists(path):
        print(f"Warning: {label} path does not exist: {path}")
        # return None # Strict check: return None
    return path # Allow non-existent path to proceed

def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM Reranking Test")
    parser.add_argument("--vlm", type=str, default="qwen", choices=["qwen", "llava"], help="VLM model type.")
    parser.add_argument("--model_name", type=str, default=None, help="Override default VLM model name.")
    parser.add_argument("--inference_backend", type=str, default="hf", choices=["hf", "vllm"], help="Inference backend ('hf' required for yesno/likert modes).")

    # --- Strategy Selection ---
    parser.add_argument("--ranking_strategy", type=str, default="pointwise",
                        choices=["pointwise", "pairwise"], # Add 'listwise' later
                        help="Ranking strategy.")

    # --- Image Inputs ---
    parser.add_argument("--query_img", type=str, default=DEFAULT_EXAMPLE_QUERY_IMG, help="Path to the query (ground-level) image.")
    parser.add_argument("--candidate_img_1", type=str, default=DEFAULT_EXAMPLE_SAT_IMG_1, help="Path to the first candidate (satellite) image.")
    parser.add_argument("--candidate_img_2", type=str, default=DEFAULT_EXAMPLE_SAT_IMG_2, help="Path to the second candidate image (required for pairwise).")
    # Add more candidate args for listwise later if needed

    # --- Prompt & Generation ---
    parser.add_argument("--prompt_mode", type=str, default="basic",
                        choices=["basic",        # Pointwise/Pairwise: Basic JSON score/preference
                                 "reasoning",    # Pointwise/Pairwise: JSON with reasoning+score/preference
                                 "yesno",        # Pointwise: Logit-based Yes/No score (HF only)
                                 "likert",       # Pointwise: Logit-based 1-5 score (HF only)
                                 "reasoning_only",# Pointwise: Generate only reasoning text (Pass 1)
                                 "reasoning_json",# Pointwise: Two-pass -> JSON score (Pass 1: reasoning, Pass 2: JSON)
                                 "reasoning_yesno"# Pointwise: Two-pass -> Yes/No score (Pass 1: reasoning, Pass 2: Yes/No)
                                ],
                        help="Prompt/Scoring mode.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens for standard generation.")
    parser.add_argument("--max_reasoning_tokens", type=int, default=512, help="Max new tokens specifically for reasoning_only pass.")


    # --- Backend/Hardware ---
    parser.add_argument("--use_quantization", action='store_true', help="Enable quantization (backend-specific method).")
    parser.add_argument("--device", type=str, default="auto", help="Device for HF backend ('auto', 'cuda', 'cpu').")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM.")

    args = parser.parse_args()

    # Add validation for HF backend requirement
    if args.prompt_mode in ["yesno", "likert", "reasoning_yesno"] and args.inference_backend != 'hf':
         parser.error(f"Prompt mode '{args.prompt_mode}' requires --inference_backend hf")
    if args.ranking_strategy != "pointwise" and args.prompt_mode not in ["basic", "reasoning"]:
         parser.error(f"Prompt mode '{args.prompt_mode}' is currently only supported for --ranking_strategy pointwise")


    return args

def parse_json_output(output_text):
    """ Robust JSON parsing from model output. """
    # Try to find JSON within ```json ... ``` or just { ... }
    match = re.search(r'```json\s*(\{.*?\})\s*```', output_text, re.DOTALL)
    if not match:
        match = re.search(r'(\{.*?\})', output_text, re.DOTALL) # Find first {} block

    if match:
        json_str = match.group(1)
        try:
            # Clean potential trailing commas before loading
            cleaned_str = json_str.rstrip(',')
            # Example: Remove trailing comma within last element if needed (more complex regex)
            # cleaned_str = re.sub(r",\s*(\}|\])$", r"\1", cleaned_str)
            parsed = json.loads(cleaned_str)
            return parsed
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse extracted JSON string: {e}")
            print(f"Extracted String: {json_str[:100]}...") # Print snippet
            return None # Indicate parsing failure
    else:
        print("Warning: No JSON object found in the output.")
        return None

def main():
    args = parse_args()

    # --- Validate and Load Images based on Strategy ---
    query_img_path = check_image_path(args.query_img, "Query Image")
    candidate_img_1_path = check_image_path(args.candidate_img_1, "Candidate Image 1")
    candidate_img_2_path = None
    images_paths_to_load = [query_img_path, candidate_img_1_path] # Paths used for loading

    if args.ranking_strategy == 'pairwise':
        print("Pairwise strategy selected. Loading second candidate image.")
        candidate_img_2_path = check_image_path(args.candidate_img_2, "Candidate Image 2")
        if not candidate_img_2_path:
            print("Error: Pairwise strategy requires --candidate_img_2 argument.")
            return
        images_paths_to_load.append(candidate_img_2_path)
    elif args.ranking_strategy == 'listwise':
        # TODO: Add logic for loading multiple candidates for listwise
        print("Error: Listwise strategy not yet implemented.")
        return

    # Ensure all necessary paths are at least provided (even if not existing yet)
    if not all(images_paths_to_load):
        print("Error: One or more required image paths are missing.")
        return

    # --- Select VLM Class ---
    vlm_class = QwenVLM if args.vlm == "qwen" else LlavaVLM
    default_model = "Qwen/Qwen2.5-VL-7B-Instruct" if args.vlm == "qwen" else "llava-hf/llava-1.5-7b-hf"
    model_name_to_load = args.model_name if args.model_name else default_model

    print(f"--- VLM Reranking Test ---")
    print(f"VLM: {args.vlm} ({model_name_to_load})")
    print(f"Backend: {args.inference_backend}, Strategy: {args.ranking_strategy}, Mode: {args.prompt_mode}")
    print(f"Device: {args.device}, Quantization: {args.use_quantization}")
    if args.inference_backend == 'vllm':
        print(f"Tensor Parallel Size: {args.tensor_parallel_size}")

    # --- Initialize VLM ---
    try:
        print(f"\nInitializing VLM...")
        vlm_instance = vlm_class(
            model_name=model_name_to_load,
            device=args.device,
            use_quantization=args.use_quantization,
            inference_backend=args.inference_backend,
            tensor_parallel_size=args.tensor_parallel_size,
            max_images_per_prompt=len(images_paths_to_load) # Inform VLM max images needed
        )
        print("VLM Initialized.")
    except Exception as e:
        print(f"FATAL: Failed to initialize VLM: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Load PIL Images ---
    images_to_pass = [] # List of actual PIL objects
    try:
        print("\nLoading images...")
        image_load_success = True
        for i, img_path in enumerate(images_paths_to_load):
            label = f"Image {i}"
            if i == 0: label = "Query"
            elif i == 1: label = "Candidate 1"
            elif i == 2: label = "Candidate 2"
            print(f" - Loading {label} from: {img_path}")
            try:
                img_obj = vlm_instance.load_image(img_path)
                images_to_pass.append(img_obj)
            except FileNotFoundError:
                 print(f"   ERROR: File not found for {label}. Cannot proceed.")
                 image_load_success = False
                 break # Stop loading if a critical image is missing
            except Exception as e:
                 print(f"   ERROR: Failed to load {label}: {e}")
                 image_load_success = False
                 break
        if not image_load_success:
             print("Aborting due to image loading errors.")
             return
        print(f"Successfully loaded {len(images_to_pass)} images.")
    except Exception as e: # Catch potential errors in the loop logic itself
        print(f"An unexpected error occurred during image loading setup: {e}")
        return

    # --- Prepare for Generation ---
    results = {}
    start_time = time.time()
    reasoning_text_cache = None # For two-pass modes

    # --- Generation Logic ---
    print(f"\n--- Running {args.ranking_strategy} - Mode: {args.prompt_mode} ---")

    try:
        # === Handle Two-Pass Modes (Pass 1: Reasoning) ===
        if args.prompt_mode in ['reasoning_json', 'reasoning_yesno']:
            print("\nRunning Pass 1: Generating Reasoning...")
            # Get the builder for reasoning_only
            reasoning_builder = get_prompt_builder(args.vlm, args.ranking_strategy, 'reasoning_only')
            prompt_conversation = reasoning_builder(mode='reasoning_only') # Pass mode explicitly

            reasoning_text_cache = vlm_instance.generate(
                conversation=prompt_conversation,
                image_inputs=images_to_pass, # Should be [query, cand1] for pointwise
                max_new_tokens=args.max_reasoning_tokens
            )
            print("-" * 20)
            print("Generated Reasoning:")
            print(reasoning_text_cache)
            print("-" * 20)
            results['pass1_reasoning'] = reasoning_text_cache

            # Now determine the mode for Pass 2
            pass2_mode = 'score_from_reasoning' if args.prompt_mode == 'reasoning_json' else 'yesno_from_reasoning'
            current_mode = pass2_mode # The mode for the actual scoring/output step
            print(f"\nRunning Pass 2: Generating Final Output (Mode: {current_mode})...")
        else:
            # Single-pass modes
            current_mode = args.prompt_mode

        # === Get Prompt Builder for the Current (or Second Pass) Step ===
        builder_func = get_prompt_builder(args.vlm, args.ranking_strategy, current_mode)
        prompt_conversation = builder_func(mode=current_mode, reasoning_text=reasoning_text_cache)

        # === Execute Generation or Scoring ===
        if current_mode in ['basic', 'reasoning', 'reasoning_only', 'score_from_reasoning']:
             # These modes use standard generate and expect text/JSON output
             output_text = vlm_instance.generate(
                 conversation=prompt_conversation,
                 image_inputs=images_to_pass,
                 max_new_tokens=args.max_new_tokens
             )
             results['raw_output'] = output_text
             print("\nModel Raw Output:")
             print(output_text)

             # Parse if JSON is expected
             if current_mode in ['basic', 'reasoning', 'score_from_reasoning'] or (args.ranking_strategy == 'pairwise' and current_mode == 'basic'):
                  parsed_output = parse_json_output(output_text)
                  results['parsed_output'] = parsed_output if parsed_output is not None else output_text # Store parsed or raw
                  if parsed_output:
                      print("\nParsed JSON Output:")
                      print(json.dumps(parsed_output, indent=2))
                  else:
                      # Already warned during parsing
                      pass


        elif current_mode in ['yesno', 'likert', 'yesno_from_reasoning']:
             # These modes use score_multiple_choice (HF backend only)
             if args.inference_backend != 'hf':
                  # This check should ideally happen in argparse, but double-check
                  raise RuntimeError(f"Mode '{current_mode}' requires HF backend.")

             choices = []
             if current_mode in ['yesno', 'yesno_from_reasoning']:
                  choices = ["Yes", "No"]
             elif current_mode == 'likert':
                  choices = ["1", "2", "3", "4", "5"]

             choice_probs = vlm_instance.score_multiple_choice(
                 conversation=prompt_conversation,
                 image_inputs=images_to_pass,
                 choices=choices
             )
             results['choice_probabilities'] = choice_probs
             print("\nChoice Probabilities:")
             print(json.dumps(choice_probs, indent=2))

             # Calculate final score based on probabilities
             if current_mode in ['yesno', 'yesno_from_reasoning']:
                  prob_yes = choice_probs.get("Yes", 0.0)
                  prob_no = choice_probs.get("No", 0.0)
                  # Normalize probabilities and calculate score (0-100)
                  if (prob_yes + prob_no) > 1e-6: # Avoid division by zero
                      score = (prob_yes / (prob_yes + prob_no)) * 100
                  else:
                      score = 50.0 # Undecided if probabilities are zero/tiny
                  results['calculated_score'] = score
                  print(f"\nCalculated Yes/No Score (0-100): {score:.2f}")

             elif current_mode == 'likert':
                   expected_value = 0.0
                   total_prob = 0.0
                   for i in range(1, 6):
                       choice_str = str(i)
                       prob = choice_probs.get(choice_str, 0.0)
                       expected_value += prob * i
                       total_prob += prob
                   # Normalize expected value (optional, but good practice if probs don't sum to 1)
                   if total_prob > 1e-6:
                        normalized_ev = expected_value / total_prob
                   else:
                        normalized_ev = 3.0 # Midpoint if probs are zero

                   # Scale 1-5 EV to 0-100 score
                   # Score = (EV - 1) * 25
                   score = (normalized_ev - 1) * 25
                   results['calculated_score'] = score
                   results['likert_expected_value'] = normalized_ev
                   print(f"\nLikert Expected Value (1-5): {normalized_ev:.2f}")
                   print(f"Calculated Likert Score (0-100): {score:.2f}")

        else:
             print(f"Error: Logic for mode '{current_mode}' not fully implemented.")
             results['error'] = f"Mode '{current_mode}' handling missing."


    except NotImplementedError as e:
        print(f"\nFATAL ERROR: Feature not implemented: {e}")
        results['error'] = str(e)
    except RuntimeError as e:
        print(f"\nFATAL RUNTIME ERROR: {e}")
        results['error'] = str(e)
        # Optional: More detailed logging
        # import traceback
        # traceback.print_exc()
    except ValueError as e: # Catch prompt builder errors etc.
         print(f"\nFATAL CONFIGURATION ERROR: {e}")
         results['error'] = str(e)
    except Exception as e:
        print(f"\nUNEXPECTED FATAL ERROR during generation: {e}")
        results['error'] = str(e)
        import traceback
        traceback.print_exc()

    # --- Final Timing and Output ---
    end_time = time.time()
    print("-" * 30)
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print("\n--- Final Results Dictionary ---")
    # Clean up results for printing (e.g., remove raw output if parsed exists)
    final_print_results = results.copy()
    if 'parsed_output' in final_print_results and final_print_results['parsed_output'] is not None:
        if 'raw_output' in final_print_results:
             del final_print_results['raw_output'] # Don't print raw if parsed succeeded

    print(json.dumps(final_print_results, indent=4))
    print("-" * 30)

if __name__ == "__main__":
    main()