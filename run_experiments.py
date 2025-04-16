import argparse
import json
import os
import time
from PIL import Image
from tqdm import tqdm # For progress bars
import traceback # For detailed error logging

# Import VLM classes and prompt builders
from models import QwenVLM, LlavaVLM # Assuming both might be used eventually
from prompts.prompt_templates import get_prompt_builder

# --- Helper Function for Safe JSON Parsing ---
def parse_vlm_output(output_text, expected_keys):
    """
    Tries to parse VLM output as JSON.
    Returns the parsed dictionary if successful and contains expected keys,
    otherwise returns the original output string.
    """
    try:
        # Clean potential markdown ```json ... ```
        if output_text.startswith("```json"):
            cleaned_text = output_text.splitlines()[1].strip()
            if cleaned_text.endswith("```"):
                 cleaned_text = cleaned_text[:-3].strip()
        elif output_text.startswith("```"): # Handle case with just ``` at start/end
             cleaned_text = output_text[3:].strip()
             if cleaned_text.endswith("```"):
                  cleaned_text = cleaned_text[:-3].strip()
        else:
             cleaned_text = output_text

        # Handle potential trailing commas causing errors
        cleaned_text = cleaned_text.rstrip(',')
        # Basic check for common mistakes if needed (e.g., replace single quotes?)
        # cleaned_text = cleaned_text.replace("'", '"') # Be careful with this

        parsed = json.loads(cleaned_text)

        if isinstance(parsed, dict) and all(key in parsed for key in expected_keys):
            return parsed # Return parsed dict
        else:
            # Keep print minimal to avoid cluttering progress bar
            # print(f"\nWarning: Parsed JSON missing expected keys ({expected_keys}). Raw output: {output_text[:50]}...")
            return output_text # Return raw text if keys mismatch

    except json.JSONDecodeError:
        # print(f"\nWarning: Failed to parse JSON. Raw output: {output_text[:50]}...")
        return output_text # Return raw text on parse failure
    except Exception as e:
        print(f"\nWarning: Unexpected error parsing output: {e}. Raw output: {output_text[:50]}...")
        return output_text

# --- Helper Function for VLM Generation Call ---
def run_vlm_generate(vlm_instance, conversation, images, max_tokens):
    """ Calls VLM generate with error handling. """
    try:
        # Basic check for image validity before calling generate
        if not images or any(img is None for img in images):
             raise ValueError("Invalid or missing image object provided.")

        output_text = vlm_instance.generate(
            conversation=conversation,
            image_inputs=images,
            max_new_tokens=max_tokens
        )
        return output_text
    except Exception as e:
        # Make error message slightly more concise for loop output
        error_msg = f"[VLM Generation Error: {type(e).__name__}]"
        print(f"\n{error_msg} (See console/log for details)")
        # Optional: Log full error details elsewhere if needed
        # traceback.print_exc()
        return error_msg # Return concise error string

# --- Main Experiment Logic ---
def main(args):
    print("--- Starting Experiment Run ---")
    print(f"Input JSON: {args.input_json}")
    print(f"Output JSON: {args.output_json}")
    print(f"VLM: {args.vlm}, Backend: {args.inference_backend}")
    exp_flags = f"Pointwise Basic={args.run_pointwise_basic}, Pointwise Reasoning={args.run_pointwise_reasoning}, Pairwise={args.run_pairwise}"
    print(f"Experiments: {exp_flags}")
    print(f"Saving results after each query to: {args.output_json}")

    # --- Load Input Data ---
    try:
        with open(args.input_json, 'r') as f:
            input_data = json.load(f)
        print(f"Loaded {len(input_data)} queries from input file.")
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {args.input_json}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse input JSON file: {args.input_json}")
        return

    # --- Initialize VLM ---
    # (VLM Initialization code remains the same as previous version)
    vlm_class = QwenVLM if args.vlm == "qwen" else LlavaVLM
    default_model = "Qwen/Qwen2.5-VL-7B-Instruct" if args.vlm == "qwen" else "llava-hf/llava-1.5-7b-hf"
    model_name_to_load = args.model_name if args.model_name else default_model
    try:
        print(f"Initializing VLM: {model_name_to_load}...")
        vlm_instance = vlm_class(
            model_name=model_name_to_load,
            device=args.device,
            use_quantization=args.use_quantization,
            inference_backend=args.inference_backend,
            tensor_parallel_size=args.tensor_parallel_size,
            max_images_per_prompt = 3 if args.run_pairwise else 2
        )
        print("VLM Initialized.")
    except Exception as e:
        print(f"FATAL: Failed to initialize VLM: {e}")
        traceback.print_exc()
        return

    # --- Prepare Prompt Builders ---
    # (Prompt Builder preparation remains the same)
    try:
        builder_pointwise = get_prompt_builder(args.vlm, 'pointwise') if args.run_pointwise_basic or args.run_pointwise_reasoning else None
        builder_pairwise = get_prompt_builder(args.vlm, 'pairwise') if args.run_pairwise else None
    except ValueError as e:
        print(f"FATAL: Error getting prompt builders: {e}")
        return

    # --- Process Queries ---
    results_data = [] # Keep results in memory
    start_run_time = time.time()

    # Check if output file exists and optionally load previous results (simple overwrite for now)
    if os.path.exists(args.output_json) and args.overwrite_output:
         print(f"Warning: Output file {args.output_json} exists and will be overwritten.")
    elif os.path.exists(args.output_json) and not args.overwrite_output:
         print(f"Error: Output file {args.output_json} already exists. Use --overwrite_output to overwrite.")
         # TODO: Implement resume logic here if desired
         return

    # Main loop processing each query item
    for item_index, item in enumerate(tqdm(input_data, desc="Processing Queries")):
        query_path = item.get("query_path")
        predictions = item.get("predictions", [])
        if not query_path or not predictions:
            print(f"\nWarning: Skipping item {item_index} due to missing query_path or predictions.")
            # Add a placeholder to results_data to keep indexing consistent if needed
            # results_data.append({"query_path": query_path, "error": "Missing data"})
            continue # Skip processing this item

        result_item = item.copy() # Start with original data
        query_img_obj = None # Load query image once per item

        # --- Load Query Image ---
        try:
            query_img_obj = vlm_instance.load_image(query_path)
        except Exception as e:
            print(f"\nFATAL Error loading query image {query_path} for item {item_index}: {e}")
            result_item["error"] = f"Failed to load query image: {e}"
            results_data.append(result_item) # Add item with error
            # Save current progress and continue to next item
            try:
                with open(args.output_json, 'w') as f:
                    json.dump(results_data, f, indent=4)
            except Exception as save_e:
                 print(f"\nError saving results after query image load failure: {save_e}")
            continue # Move to next item in input_data

        # --- Experiment 1: Pointwise Basic ---
        if args.run_pointwise_basic:
            pointwise_basic_scores = []
            # Use tqdm for the inner loop as well
            for cand_path in tqdm(predictions, desc=f"Query {item_index+1} PW Basic", leave=False):
                try:
                    cand_img = vlm_instance.load_image(cand_path)
                except Exception as e:
                    print(f"\nError loading candidate image {cand_path}: {e}")
                    pointwise_basic_scores.append("[Image Load Error]")
                    continue

                conversation = builder_pointwise(mode='basic')
                output_text = run_vlm_generate(vlm_instance, conversation, [query_img_obj, cand_img], args.max_new_tokens)
                parsed = parse_vlm_output(output_text, expected_keys=['score'])
                pointwise_basic_scores.append(parsed.get('score') if isinstance(parsed, dict) else parsed)
            result_item['pointwise_basic_scores'] = pointwise_basic_scores

        # --- Experiment 2: Pointwise Reasoning ---
        if args.run_pointwise_reasoning:
            pointwise_reasoning_scores = []
            pointwise_reasoning_texts = []
            for cand_path in tqdm(predictions, desc=f"Query {item_index+1} PW Reason", leave=False):
                try:
                    cand_img = vlm_instance.load_image(cand_path)
                except Exception as e:
                    print(f"\nError loading candidate image {cand_path}: {e}")
                    pointwise_reasoning_scores.append(None)
                    pointwise_reasoning_texts.append("[Image Load Error]")
                    continue

                conversation = builder_pointwise(mode='reasoning')
                output_text = run_vlm_generate(vlm_instance, conversation, [query_img_obj, cand_img], args.max_new_tokens)
                parsed = parse_vlm_output(output_text, expected_keys=['score', 'reasoning'])
                if isinstance(parsed, dict):
                    pointwise_reasoning_scores.append(parsed.get('score'))
                    pointwise_reasoning_texts.append(parsed.get('reasoning'))
                else:
                    pointwise_reasoning_scores.append(None)
                    pointwise_reasoning_texts.append(parsed)
            result_item['pointwise_reasoning_scores'] = pointwise_reasoning_scores
            result_item['pointwise_reasoning_texts'] = pointwise_reasoning_texts

        # --- Experiment 3: Pairwise Top-1 ---
        if args.run_pairwise:
            pairwise_top1_prediction = "[No Predictions]" # Default if predictions list is empty
            if predictions: # Only run if there are predictions
                current_best_path = predictions[0]
                for i in tqdm(range(1, len(predictions)), desc=f"Query {item_index+1} Pairwise", leave=False):
                    candidate_path_1 = current_best_path
                    candidate_path_2 = predictions[i]
                    cand1_img, cand2_img = None, None # Reset images for this pair

                    try:
                        # Query image already loaded (query_img_obj)
                        cand1_img = vlm_instance.load_image(candidate_path_1)
                        cand2_img = vlm_instance.load_image(candidate_path_2)
                    except Exception as e:
                        print(f"\nError loading images for Pairwise comp {i}: {e}")
                        print(f"Skipping comparison involving: {os.path.basename(candidate_path_1)}, {os.path.basename(candidate_path_2)}")
                        # Keep current_best_path as is and continue to next comparison
                        continue

                    conversation = builder_pairwise(mode='basic') # Use basic for preference only
                    output_text = run_vlm_generate(vlm_instance, conversation, [query_img_obj, cand1_img, cand2_img], args.max_new_tokens)
                    parsed = parse_vlm_output(output_text, expected_keys=['preference'])

                    if isinstance(parsed, dict):
                        preference = parsed.get('preference')
                        try:
                           pref_int = int(preference)
                           if pref_int == 2:
                               current_best_path = candidate_path_2
                           elif pref_int != 1: # Only update if preference is explicitly 2
                                pass # Keep current best if pref is 1 or invalid
                        except (ValueError, TypeError):
                             # Keep current best if preference is not easily convertible to 1 or 2
                             pass # Logged by parse_vlm_output if needed
                    else:
                        # Keep current best if parsing failed or VLM error occurred
                        # Error/warning already printed by parse_vlm_output or run_vlm_generate
                         pass

                pairwise_top1_prediction = current_best_path
            result_item['pairwise_top1_prediction'] = pairwise_top1_prediction

        # --- Append result for this query and SAVE ---
        results_data.append(result_item)

        # Save the entire results list after processing this query item
        try:
            with open(args.output_json, 'w') as f:
                 json.dump(results_data, f, indent=4)
        except Exception as e:
            print(f"\nError saving results to {args.output_json} after processing query {item_index}: {e}")
            # Decide how to handle this - maybe try saving to a backup file?
            # For now, just print error and continue. Data for this query is lost if script crashes now.


    # --- Final Summary ---
    end_run_time = time.time()
    print("-" * 30)
    print(f"Finished processing {len(input_data)} queries.")
    print(f"Total runtime: {end_run_time - start_run_time:.2f} seconds.")
    print(f"Final results saved to {args.output_json}") # File was updated iteratively
    print("--- Experiment Run Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM Reranking Experiments")
    # Using os.path.join for potentially better path handling across OS
    DEFAULT_JSON_PATH = '/research/nfs_yilmaz_15/visual_reranker/Sample4Geo/results/vigor/cross_test_sampled_results.json'
    DEFAULT_OUTPUT_PATH = '/research/nfs_yilmaz_15/yunus/geo_ranker/cross_test_results.json'

    # --- Input/Output ---
    parser.add_argument("--input_json", type=str, default=DEFAULT_JSON_PATH, help="Path to input JSON file.")
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to save output JSON file.")
    # Remove save_interval argument
    # parser.add_argument("--save_interval", type=int, default=50, help="Save intermediate results every N queries.")
    parser.add_argument("--overwrite_output", action='store_true', help="Overwrite the output file if it exists.")


    # --- Model & Backend ---
    # (Arguments remain the same)
    parser.add_argument("--vlm", type=str, default="qwen", choices=["qwen", "llava"], help="VLM model type.")
    parser.add_argument("--model_name", type=str, default=None, help="Override default VLM model name.")
    parser.add_argument("--inference_backend", type=str, default="vllm", choices=["hf", "vllm"], help="Inference backend.")
    parser.add_argument("--use_quantization", action='store_true', help="Enable quantization.")
    parser.add_argument("--device", type=str, default="auto", help="Device for HF backend.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM.")

    # --- Experiment Selection ---
    # (Arguments remain the same)
    parser.add_argument("--run_pointwise_basic", action='store_true', help="Run Pointwise Basic scoring experiment.")
    parser.add_argument("--run_pointwise_reasoning", action='store_true', help="Run Pointwise Reasoning scoring experiment.")
    parser.add_argument("--run_pairwise", action='store_true', help="Run Pairwise Top-1 reranking experiment.")

    # --- Generation Params ---
    # (Arguments remain the same)
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens for generation.")

    parsed_args = parser.parse_args()
    if not any([parsed_args.run_pointwise_basic, parsed_args.run_pointwise_reasoning, parsed_args.run_pairwise]):
        parser.error("No experiments selected. Please enable at least one flag (e.g., --run_pointwise_basic).")

    main(parsed_args)