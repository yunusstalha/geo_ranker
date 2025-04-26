import argparse
import json
import os
import time
from PIL import Image
from tqdm import tqdm 
import traceback 
import numpy as np 

# Import VLM classes and prompt builders
from models import QwenVLM, LlavaVLM
# Use the getter function which now handles modes correctly
from prompts.prompt_templates import get_prompt_builder

# --- Helper Function for Safe JSON Parsing ---
def parse_vlm_output(output_text, expected_keys):
    """
    Tries to parse VLM output as JSON. Returns parsed dict or raw text.
    Improved robustness.
    """
    if not isinstance(output_text, str): # Handle non-string inputs (e.g., errors)
        return output_text

    try:
        cleaned_text = output_text.strip()
        # Remove potential markdown fences
        if cleaned_text.startswith("```json") and cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[len("```json"):-len("```")].strip()
        elif cleaned_text.startswith("```") and cleaned_text.endswith("```"):
             cleaned_text = cleaned_text[len("```"):-len("```")].strip()

        # Find the first '{' and last '}' to extract potential JSON object
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
             json_str = cleaned_text[start_idx : end_idx + 1]
             # Attempt parsing
             parsed = json.loads(json_str)

             if isinstance(parsed, dict) and all(key in parsed for key in expected_keys):
                 return parsed # Return parsed dict
             else:
                 # print(f"\nWarning: Parsed JSON missing expected keys ({expected_keys}). Raw: {json_str[:100]}...")
                 return output_text # Return raw text if keys mismatch
        else:
            # No JSON object structure found
             return output_text

    except json.JSONDecodeError:
        # print(f"\nWarning: Failed to parse JSON. Raw output snippet: {output_text[:50]}...")
        return output_text # Return raw text on parse failure
    except Exception as e:
        print(f"\nWarning: Unexpected error parsing output: {e}. Raw: {output_text[:100]}...")
        return output_text

# --- Helper Function for VLM Standard Generation Call ---
def run_vlm_generate(vlm_instance, conversation, images, max_tokens):
    """ Calls VLM generate with error handling. Returns output string or error string. """
    try:
        if not images or any(img is None for img in images):
             raise ValueError("Invalid or missing image object provided.")
        # Assuming VLM classes have @torch.no_grad() decorator on _generate_hf
        output_text = vlm_instance.generate(
            conversation=conversation, image_inputs=images, max_new_tokens=max_tokens
        )
        # Check if output itself indicates an error (e.g., from vLLM backend)
        if isinstance(output_text, str) and "[VLM generation failed]" in output_text:
             print(f"\nWarning: VLM generate returned failure message: {output_text}")
             return f"[VLM Generate Failed: {output_text}]"
        return output_text
    except NotImplementedError as nie:
        error_msg = f"[VLM Generate Not Impl Error: {nie}]"
        print(f"\n{error_msg}")
        return error_msg
    except RuntimeError as rte:
        error_msg = f"[VLM Generate Runtime Error: {rte}]"
        print(f"\n{error_msg}")
        # traceback.print_exc() # Optionally uncomment for more detail
        return error_msg
    except Exception as e:
        error_msg = f"[VLM Generation Error: {type(e).__name__}]"
        print(f"\n{error_msg} (Check logs)")
        traceback.print_exc() # Log full traceback for unexpected errors
        return error_msg


# --- Helper Function for VLM Score Multiple Choice Call ---
def run_vlm_score_mc(vlm_instance, conversation, images, choices):
    """ Calls VLM score_multiple_choice with error handling. Returns dict of probs or dict with error."""
    try:
        if not images or any(img is None for img in images):
             raise ValueError("Invalid or missing image object provided.")
        # Assumes score_multiple_choice already has @torch.no_grad()
        choice_probs = vlm_instance.score_multiple_choice(
            conversation=conversation, image_inputs=images, choices=choices
        )
        return choice_probs
    except NotImplementedError as nie:
        error_msg = f"[VLM Score MC Not Impl Error: {nie}]"
        print(f"\n{error_msg}")
        return {"error": error_msg}
    except RuntimeError as rte:
        error_msg = f"[VLM Score MC Runtime Error: {rte}]"
        print(f"\n{error_msg}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"[VLM Score MC Error: {type(e).__name__}]"
        print(f"\n{error_msg} (Check logs)")
        traceback.print_exc()
        return {"error": error_msg}


# --- Main Experiment Logic ---
def main(args):
    print("--- Starting Experiment Run ---")
    print(f"Input JSON: {args.input_json}")
    print(f"Output JSON: {args.output_json}")
    print(f"VLM: {args.vlm}, Backend: {args.inference_backend}")
    # Update experiments printout
    exp_flags = []
    if args.run_pointwise_basic: exp_flags.append("PW_Basic")
    if args.run_pointwise_reasoning: exp_flags.append("PW_Reasoning")
    if args.run_pointwise_yesno: exp_flags.append("PW_YesNo")
    if args.run_pointwise_likert: exp_flags.append("PW_Likert")
    if args.run_pointwise_reasoning_yesno: exp_flags.append("PW_ReasonYesNo")
    if args.run_pairwise_basic: exp_flags.append("PairwiseBasic")
    if args.run_pairwise_reasoning: exp_flags.append("PairwiseReasoning2Pass")
    print(f"Experiments: {', '.join(exp_flags)}")
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
    vlm_class = QwenVLM if args.vlm == "qwen" else LlavaVLM
    default_model = "Qwen/Qwen2.5-VL-7B-Instruct" if args.vlm == "qwen" else "llava-hf/llava-1.5-7b-hf"
    model_name_to_load = args.model_name if args.model_name else default_model
    try:
        print(f"\nInitializing VLM: {model_name_to_load}...")
        max_images = 2
        if args.run_pairwise_basic or args.run_pairwise_reasoning: # Check args directly
             max_images = 3
        vlm_instance = vlm_class(
            model_name=model_name_to_load, device=args.device, use_quantization=args.use_quantization,
            inference_backend=args.inference_backend, tensor_parallel_size=args.tensor_parallel_size,
            max_images_per_prompt = max_images
        )
        print("VLM Initialized.")
    except Exception as e:
        print(f"FATAL: Failed to initialize VLM: {e}")
        traceback.print_exc()
        return

    # --- Prepare Prompt Builders ---
    builder_pointwise = None
    builder_pairwise = None
    try:
        if any([args.run_pointwise_basic, args.run_pointwise_reasoning,
                args.run_pointwise_yesno, args.run_pointwise_likert,
                args.run_pointwise_reasoning_yesno]):
             builder_pointwise = get_prompt_builder(args.vlm, 'pointwise') # Get base pointwise builder

        if args.run_pairwise_basic or args.run_pairwise_reasoning:
            builder_pairwise = get_prompt_builder(args.vlm, 'pairwise') # Get base pairwise builder

    except ValueError as e:
        print(f"FATAL: Error getting prompt builders: {e}")
        return

    # --- Process Queries ---
    results_data = []
    start_run_time = time.time()

    # Output file handling
    if os.path.exists(args.output_json) and args.overwrite_output:
         print(f"Warning: Output file {args.output_json} exists and will be overwritten.")
    elif os.path.exists(args.output_json) and not args.overwrite_output:
         print(f"Error: Output file {args.output_json} already exists. Use --overwrite_output to overwrite.")
         return

    # Main loop processing each query item
    for item_index, item in enumerate(tqdm(input_data, desc="Processing Queries")):
        query_path = item.get("query_path")
        predictions = item.get("predictions", [])
        if not query_path or not predictions:
            print(f"\nWarning: Skipping item {item_index} due to missing 'query_path' or empty 'predictions'.")
            continue

        result_item = item.copy()
        query_img_obj = None

        # --- Load Query Image ---
        try:
            query_img_obj = vlm_instance.load_image(query_path)
        except Exception as e:
            print(f"\nError loading query image {query_path} for item {item_index}: {e}")
            result_item["query_error"] = f"Failed to load query image: {e}"
            results_data.append(result_item)
            try: # Try to save partial results
                with open(args.output_json, 'w') as f: json.dump(results_data, f, indent=4, default=str)
            except Exception as save_e: print(f"\nError saving results after query image load failure: {save_e}")
            continue # Skip to next query item

        # --- Initialize results lists for this query item ---
        if args.run_pointwise_basic: result_item['pointwise_basic_results'] = []
        if args.run_pointwise_reasoning: result_item['pointwise_reasoning_results'] = []
        if args.run_pointwise_yesno: result_item['pointwise_yesno_results'] = []
        if args.run_pointwise_likert: result_item['pointwise_likert_results'] = []
        if args.run_pointwise_reasoning_yesno: result_item['pointwise_reasoning_yesno_results'] = []
        # Initialize pairwise keys based on args
        if args.run_pairwise_basic:
            result_item['pairwise_basic_comparisons'] = []
            result_item['pairwise_basic_ranking'] = []
        if args.run_pairwise_reasoning:
            result_item['pairwise_reasoning_comparisons'] = []
            result_item['pairwise_reasoning_ranking'] = []

        # Determine if any pairwise mode is active for the loop later
        run_pairwise_active = args.run_pairwise_basic or args.run_pairwise_reasoning

        # --- Loop through Candidate Predictions for Pointwise Modes ---
        # This loop populates the pointwise results if flags are set
        if any([args.run_pointwise_basic, args.run_pointwise_reasoning,
                args.run_pointwise_yesno, args.run_pointwise_likert,
                args.run_pointwise_reasoning_yesno]):

            for cand_idx, cand_path in enumerate(tqdm(predictions, desc=f"Query {item_index+1} PW Cands", leave=False)):
                cand_img = None
                # Append error placeholder function
                def append_error(mode_key, error_msg="Image Load Error"):
                     if mode_key in result_item:
                          result_item[mode_key].append({"error": error_msg})

                try:
                    cand_img = vlm_instance.load_image(cand_path)
                    pointwise_images = [query_img_obj, cand_img] # Images needed for pointwise
                except Exception as e:
                    print(f"\nError loading candidate image {cand_path} (Cand {cand_idx}) for Pointwise: {e}")
                    # Add error placeholders to all active pointwise modes
                    if args.run_pointwise_basic: append_error('pointwise_basic_results')
                    if args.run_pointwise_reasoning: append_error('pointwise_reasoning_results')
                    if args.run_pointwise_yesno: append_error('pointwise_yesno_results')
                    if args.run_pointwise_likert: append_error('pointwise_likert_results')
                    if args.run_pointwise_reasoning_yesno: append_error('pointwise_reasoning_yesno_results')
                    continue # Skip this candidate for pointwise

                # --- Pointwise Basic ---
                if args.run_pointwise_basic and builder_pointwise:
                    conversation = builder_pointwise(mode='basic')
                    output_text = run_vlm_generate(vlm_instance, conversation, pointwise_images, args.max_new_tokens)
                    parsed = parse_vlm_output(output_text, expected_keys=['score'])
                    result_item['pointwise_basic_results'].append(parsed if isinstance(parsed, dict) else {"raw_output": output_text, "error": "Parsing failed or keys missing"})

                # --- Pointwise Reasoning (Single Pass) ---
                if args.run_pointwise_reasoning and builder_pointwise:
                     conversation = builder_pointwise(mode='reasoning')
                     output_text = run_vlm_generate(vlm_instance, conversation, pointwise_images, args.max_new_tokens)
                     parsed = parse_vlm_output(output_text, expected_keys=['score', 'reasoning'])
                     result_item['pointwise_reasoning_results'].append(parsed if isinstance(parsed, dict) else {"raw_output": output_text, "error": "Parsing failed or keys missing"})

                # --- Pointwise Yes/No ---
                if args.run_pointwise_yesno and builder_pointwise:
                    # Requires HF backend, checked at arg parsing
                    yesno_choices = ["Yes", "No"]
                    conversation = builder_pointwise(mode='yesno')
                    choice_probs = run_vlm_score_mc(vlm_instance, conversation, pointwise_images, yesno_choices)
                    score = None
                    error_msg = choice_probs.get("error")
                    if not error_msg:
                        prob_yes = choice_probs.get("Yes", 0.0)
                        prob_no = choice_probs.get("No", 0.0)
                        denominator = prob_yes + prob_no
                        if denominator > 1e-9: score = (prob_yes / denominator) * 100
                        else: score = 50.0 # Undecided
                    result_item['pointwise_yesno_results'].append({
                        "probabilities": choice_probs,
                        "calculated_score": score if not error_msg else None,
                        "error": error_msg
                    })

                # --- Pointwise Likert ---
                if args.run_pointwise_likert and builder_pointwise:
                    # Requires HF backend, checked at arg parsing
                    likert_choices = ["1", "2", "3", "4", "5"]
                    conversation = builder_pointwise(mode='likert')
                    choice_probs = run_vlm_score_mc(vlm_instance, conversation, pointwise_images, likert_choices)
                    score = None
                    expected_value = None
                    error_msg = choice_probs.get("error")
                    if not error_msg:
                        ev_sum, prob_sum = 0.0, 0.0
                        for i_str in likert_choices:
                            prob = choice_probs.get(i_str, 0.0)
                            ev_sum += prob * int(i_str)
                            prob_sum += prob
                        if prob_sum > 1e-9: expected_value = ev_sum / prob_sum
                        else: expected_value = 3.0 # Midpoint if no probability mass
                        score = (expected_value - 1) * 25 # Scale 1-5 to 0-100
                    result_item['pointwise_likert_results'].append({
                        "probabilities": choice_probs,
                        "expected_value": expected_value if not error_msg else None,
                        "calculated_score": score if not error_msg else None,
                        "error": error_msg
                    })

                # --- Pointwise Reasoning + Yes/No (Two-Pass) ---
                if args.run_pointwise_reasoning_yesno and builder_pointwise:
                     # Requires HF backend, checked at arg parsing
                     reasoning_text = "[Error in Pass 1]"
                     choice_probs = {"error": "Pass 1 Failed"}
                     score = None
                     error_msg_pass1 = None
                     error_msg_pass2 = None

                     # Pass 1: Generate Reasoning
                     try:
                         conv_pass1 = builder_pointwise(mode='reasoning_only')
                         reasoning_text = run_vlm_generate(vlm_instance, conv_pass1, pointwise_images, args.max_reasoning_tokens)
                         if reasoning_text.startswith("[VLM"):
                              error_msg_pass1 = reasoning_text
                              reasoning_text = None # Prevent Pass 2
                     except Exception as e1:
                         error_msg_pass1 = f"[Pass 1 Exception: {type(e1).__name__}]"
                         print(f"\n{error_msg_pass1} for PW cand {cand_idx}")
                         # traceback.print_exc()
                         reasoning_text = None # Prevent Pass 2

                     # Pass 2: Get Yes/No Score (only if Pass 1 didn't error)
                     if reasoning_text is not None:
                         try:
                             yesno_choices = ["Yes", "No"]
                             conv_pass2 = builder_pointwise(mode='yesno_from_reasoning', reasoning_text=reasoning_text)
                             choice_probs = run_vlm_score_mc(vlm_instance, conv_pass2, pointwise_images, yesno_choices)
                             error_msg_pass2 = choice_probs.get("error")
                             if not error_msg_pass2:
                                 prob_yes = choice_probs.get("Yes", 0.0)
                                 prob_no = choice_probs.get("No", 0.0)
                                 denominator = prob_yes + prob_no
                                 if denominator > 1e-9: score = (prob_yes / denominator) * 100
                                 else: score = 50.0 # Undecided
                         except Exception as e2:
                              error_msg_pass2 = f"[Pass 2 Exception: {type(e2).__name__}]"
                              print(f"\n{error_msg_pass2} for PW cand {cand_idx}")
                              # traceback.print_exc()
                              choice_probs = {"error": error_msg_pass2}

                     # Store combined results for this candidate
                     result_item['pointwise_reasoning_yesno_results'].append({
                         "reasoning": reasoning_text if error_msg_pass1 is None else error_msg_pass1,
                         "probabilities": choice_probs,
                         "calculated_score": score if not error_msg_pass1 and not error_msg_pass2 else None,
                         "error": error_msg_pass1 or error_msg_pass2
                     })

                # Clean up candidate image from memory
                del cand_img


        # PAIRWISE SECTION
        if run_pairwise_active and builder_pairwise:
            # Determine the specific mode ('basic' or 'reasoning') based on args
            run_pairwise_mode_type = 'basic' if args.run_pairwise_basic else 'reasoning'
            pairwise_mode_label = "Basic" if run_pairwise_mode_type == 'basic' else "Reasoning (2-Pass)"
            print(f"\n-- Running Pairwise Sorting (Mode: {pairwise_mode_label}, Top K: {args.pairwise_top_k}) --")

            # Work on a copy of predictions to sort in place for this item
            candidate_paths = list(predictions) # Use the original list
            num_candidates = len(candidate_paths)
            comparisons_log = [] # Log individual comparisons

            # Selection Sort for Top-K
            k = min(args.pairwise_top_k, num_candidates)
            for i in range(k):
                current_best_index = i
                # Compare candidate[i] with subsequent candidates
                for j in range(i + 1, num_candidates):
                    cand1_path = candidate_paths[current_best_index]
                    cand2_path = candidate_paths[j]
                    cand1_img, cand2_img = None, None

                    # Initialize comparison entry
                    comparison_entry = {
                        "comparison_index": len(comparisons_log),
                        "cand1_path": cand1_path,
                        "cand2_path": cand2_path,
                        "current_best_before_swap": candidate_paths[i],
                        "winner_index_in_comparison": 1, # Default assumption
                        "reasoning": None, # For reasoning mode
                        "preference": None, # Final preference
                        # "confidence": None, # REMOVED
                        "pass1_raw_output": None,
                        "pass2_raw_output": None, # Only used in 2-pass reasoning
                        "error_pass1": None,
                        "error_pass2": None, # Only used in 2-pass reasoning
                        "combined_error": None
                    }

                    try:
                        # --- Load Images ---
                        cand1_img = vlm_instance.load_image(cand1_path)
                        cand2_img = vlm_instance.load_image(cand2_path)
                        images_for_vlm = [query_img_obj, cand1_img, cand2_img]

                        # --- Execute VLM Call(s) based on mode ---
                        if run_pairwise_mode_type == 'basic':
                            # --- Basic Mode (Single Pass) ---
                            conv_pass1 = builder_pairwise(mode='basic')
                            output_pass1 = run_vlm_generate(
                                vlm_instance, conv_pass1, images_for_vlm, args.max_new_tokens
                            )
                            comparison_entry["pass1_raw_output"] = output_pass1

                            if output_pass1.startswith("[VLM"): # Check for generate error
                                comparison_entry["error_pass1"] = output_pass1
                            else:
                                parsed = parse_vlm_output(output_pass1, expected_keys=['preference']) # Only expect preference
                                if isinstance(parsed, dict):
                                    preference = parsed.get('preference')
                                    comparison_entry["preference"] = preference
                                    try:
                                        if int(preference) == 2:
                                            current_best_index = j
                                            comparison_entry["winner_index_in_comparison"] = 2
                                    except (ValueError, TypeError):
                                        comparison_entry["error_pass1"] = "Invalid preference value in JSON (Basic)"
                                else:
                                    comparison_entry["error_pass1"] = "Parsing failed or keys missing (Basic)"
                                    # comparison_entry["pass1_raw_output"] = parsed # Already stored raw output

                        elif run_pairwise_mode_type == 'reasoning':
                            # --- Reasoning Mode (Two Pass - No Confidence) ---
                            reasoning_text = None
                            preference = None

                            # --- Pass 1: Generate Reasoning ---
                            try:
                                conv_pass1 = builder_pairwise(mode='reasoning_only')
                                reasoning_text = run_vlm_generate(
                                    vlm_instance, conv_pass1, images_for_vlm, args.max_reasoning_tokens
                                )
                                comparison_entry["pass1_raw_output"] = reasoning_text
                                comparison_entry["reasoning"] = reasoning_text # Store reasoning

                                if reasoning_text.startswith("[VLM"): # Check helper error
                                    comparison_entry["error_pass1"] = reasoning_text
                                    reasoning_text = None # Prevent Pass 2

                            except Exception as e1_pairwise:
                                err_msg = f"[Pass 1 VLM Error: {type(e1_pairwise).__name__}]"
                                print(f"\n{err_msg} for comparison {len(comparisons_log)}")
                                # traceback.print_exc()
                                comparison_entry["error_pass1"] = err_msg
                                reasoning_text = None # Prevent Pass 2

                            # --- Pass 2: Get Preference from Reasoning ---
                            if reasoning_text is not None and not comparison_entry["error_pass1"]: # Only run if Pass 1 succeeded
                                try:
                                    conv_pass2 = builder_pairwise(mode='preference_from_reasoning', reasoning_text=reasoning_text)
                                    output_pass2 = run_vlm_generate(
                                        vlm_instance, conv_pass2, images_for_vlm, args.max_new_tokens
                                    )
                                    comparison_entry["pass2_raw_output"] = output_pass2

                                    if output_pass2.startswith("[VLM"): # Check helper error
                                        comparison_entry["error_pass2"] = output_pass2
                                    else:
                                        # MODIFIED: Only expect 'preference'
                                        parsed = parse_vlm_output(output_pass2, expected_keys=['preference'])
                                        if isinstance(parsed, dict):
                                            preference = parsed.get('preference')
                                            comparison_entry["preference"] = preference
                                            # comparison_entry["confidence"] = parsed.get('confidence') # REMOVED
                                            try:
                                                if int(preference) == 2:
                                                    current_best_index = j
                                                    comparison_entry["winner_index_in_comparison"] = 2
                                            except (ValueError, TypeError):
                                                comparison_entry["error_pass2"] = "Invalid preference value in JSON (Pass 2)"
                                        else:
                                            comparison_entry["error_pass2"] = "Parsing failed or key missing (Pass 2)"
                                            # comparison_entry["pass2_raw_output"] = parsed # Already stored raw

                                except Exception as e2_pairwise:
                                     err_msg = f"[Pass 2 VLM Error: {type(e2_pairwise).__name__}]"
                                     print(f"\n{err_msg} for comparison {len(comparisons_log)}")
                                     # traceback.print_exc()
                                     comparison_entry["error_pass2"] = err_msg

                    except Exception as img_load_err:
                        # Catch errors during image loading
                        err_msg = f"Error loading images for pairwise comparison: {type(img_load_err).__name__}"
                        print(f"\n{err_msg}")
                        comparison_entry["combined_error"] = err_msg
                        # Assume cand1 wins if images can't be loaded for comparison

                    finally:
                        # Combine errors and clean up
                        comparison_entry["combined_error"] = comparison_entry["error_pass1"] or comparison_entry["error_pass2"] or comparison_entry["combined_error"]
                        # Clean up individual error fields if desired, or keep for debugging
                        # if comparison_entry["combined_error"]:
                        #     if "error_pass1" in comparison_entry: del comparison_entry["error_pass1"]
                        #     if "error_pass2" in comparison_entry: del comparison_entry["error_pass2"]

                        del cand1_img, cand2_img # Free memory
                        comparisons_log.append(comparison_entry)
                        # time.sleep(0.1) # Optional delay

                # --- End of inner comparison loop (j) ---

                # After comparing candidate[i] with all subsequent ones, swap the best found into position i
                if current_best_index != i:
                    candidate_paths[i], candidate_paths[current_best_index] = candidate_paths[current_best_index], candidate_paths[i]

            # --- End of outer sorting loop (i) ---

            # Store results for the correct mode
            if args.run_pairwise_basic:
                 result_item['pairwise_basic_comparisons'] = comparisons_log
                 result_item['pairwise_basic_ranking'] = candidate_paths[:k]
            elif args.run_pairwise_reasoning: # This flag now implies 2-pass
                 result_item['pairwise_reasoning_comparisons'] = comparisons_log
                 result_item['pairwise_reasoning_ranking'] = candidate_paths[:k]

        # <<< --- PAIRWISE SECTION END --- >>>


        # --- Append result for this query and SAVE ---
        results_data.append(result_item)
        try:
            with open(args.output_json, 'w') as f:
                 # Use default=str to handle potential non-serializable types like numpy floats
                 json.dump(results_data, f, indent=4, default=str)
        except Exception as e:
            print(f"\nError saving results to {args.output_json} after processing query {item_index}: {e}")


    # --- Final Summary ---
    end_run_time = time.time()
    print("-" * 30)
    print(f"Finished processing {len(results_data)} / {len(input_data)} queries.")
    print(f"Total runtime: {end_run_time - start_run_time:.2f} seconds.")
    print(f"Final results saved to {args.output_json}")
    print("--- Experiment Run Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM Reranking Experiments")
    # Define paths
    DEFAULT_JSON_PATH = '/research/nfs_yilmaz_15/visual_reranker/Sample4Geo/results/vigor/cross_test_sampled_results.json'
    DEFAULT_OUTPUT_PATH_BASE = '/research/nfs_yilmaz_15/yunus/geo_ranker/exp_results' # Base dir for output

    # --- Input/Output ---
    parser.add_argument("--input_json", type=str, default=DEFAULT_JSON_PATH, help="Path to input JSON file.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_PATH_BASE, help="Directory to save output JSON file.")
    parser.add_argument("--exp_name", type=str, default=None, help="Optional experiment name for output filename.")
    parser.add_argument("--overwrite_output", action='store_true', help="Overwrite the output file if it exists.")

    # --- Model & Backend ---
    parser.add_argument("--vlm", type=str, required=True, choices=["qwen", "llava"], help="VLM model type.")
    parser.add_argument("--model_name", type=str, default=None, help="Override default VLM model name.")
    parser.add_argument("--inference_backend", type=str, default="hf", choices=["hf", "vllm"], help="Inference backend.")
    parser.add_argument("--use_quantization", action='store_true', help="Enable quantization.")
    parser.add_argument("--device", type=str, default="auto", help="Device for HF backend ('auto', 'cuda', 'cpu').")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM.")

    # --- Experiment Selection ---
    parser.add_argument("--run_pointwise_basic", action='store_true', help="Run Pointwise Basic scoring.")
    parser.add_argument("--run_pointwise_reasoning", action='store_true', help="Run Pointwise Reasoning scoring (1-Pass).")
    parser.add_argument("--run_pointwise_yesno", action='store_true', help="Run Pointwise Yes/No scoring (HF only).")
    parser.add_argument("--run_pointwise_likert", action='store_true', help="Run Pointwise Likert scoring (HF only).")
    parser.add_argument("--run_pointwise_reasoning_yesno", action='store_true', help="Run 2-Pass Pointwise: Reasoning -> Yes/No score (HF only).")
    parser.add_argument("--run_pairwise_basic", action='store_true', help="Run Pairwise reranking (Basic Preference, 1-Pass).")
    parser.add_argument("--run_pairwise_reasoning", action='store_true', help="Run Pairwise reranking (Reasoning -> Preference, 2-Pass).") # Updated help
    parser.add_argument("--pairwise_top_k", type=int, default=3, help="Number of top candidates to find using pairwise sorting.")

    # --- Generation Params ---
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens for JSON/text generation (Pass 2 for 2-pass).")
    parser.add_argument("--max_reasoning_tokens", type=int, default=256, help="Max new tokens for reasoning_only pass (Pass 1 for 2-pass modes).")


    parsed_args = parser.parse_args()

    # --- Validate Args ---
    experiment_flags = [
        parsed_args.run_pointwise_basic, parsed_args.run_pointwise_reasoning,
        parsed_args.run_pointwise_yesno, parsed_args.run_pointwise_likert,
        parsed_args.run_pointwise_reasoning_yesno,
        parsed_args.run_pairwise_basic, parsed_args.run_pairwise_reasoning,
    ]
    if not any(experiment_flags):
        parser.error("No experiments selected. Please enable at least one run flag (e.g., --run_pairwise_reasoning).")

    # HF backend check for specific modes
    if (parsed_args.run_pointwise_yesno or parsed_args.run_pointwise_likert or parsed_args.run_pointwise_reasoning_yesno) and parsed_args.inference_backend != 'hf':
         parser.error("--run_pointwise_yesno, --run_pointwise_likert, and --run_pointwise_reasoning_yesno require --inference_backend hf")

    # --- Construct Output Path ---
    os.makedirs(parsed_args.output_dir, exist_ok=True)
    output_filename_parts = [parsed_args.vlm]
    if parsed_args.model_name:
         # Sanitize model name for filename
         sanitized_model_name = os.path.basename(parsed_args.model_name).replace('/', '_')
         output_filename_parts.append(sanitized_model_name)
    if parsed_args.exp_name:
         output_filename_parts.append(parsed_args.exp_name)
    else:
        # Auto-generate name based on active modes
        active_modes = []
        if parsed_args.run_pointwise_basic: active_modes.append("pw_basic")
        if parsed_args.run_pointwise_reasoning: active_modes.append("pw_reason")
        if parsed_args.run_pointwise_yesno: active_modes.append("pw_yesno")
        if parsed_args.run_pointwise_likert: active_modes.append("pw_likert")
        if parsed_args.run_pointwise_reasoning_yesno: active_modes.append("pw_reason_yesno")
        if parsed_args.run_pairwise_basic: active_modes.append("pair_basic")
        if parsed_args.run_pairwise_reasoning: active_modes.append("pair_reason2pass") # Use specific tag
        output_filename_parts.append("_".join(active_modes) if active_modes else "no_modes")

    output_filename = "_".join(output_filename_parts) + ".json"
    parsed_args.output_json = os.path.join(parsed_args.output_dir, output_filename)

    main(parsed_args)