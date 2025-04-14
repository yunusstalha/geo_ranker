import argparse
import torch
from PIL import Image
import json
import time
import os

# Import VLM classes and prompt builders
from models import QwenVLM, LlavaVLM
# Import the new helper function from prompt_templates
from prompts.prompt_templates import get_prompt_builder

# --- Example Image Paths ---
DEFAULT_EXAMPLE_QUERY_IMG = "/research/nfs_yilmaz_15/yunus/data/VIGOR/SanFrancisco/panorama/ScGwgVhnS9mLGl47wRnicQ,37.752695,-122.437849,.jpg"
DEFAULT_EXAMPLE_SAT_IMG_1 = "/research/nfs_yilmaz_15/yunus/data/VIGOR/SanFrancisco/satellite/satellite_37.752731622631885_-122.43810708310743.png" # Incorrect match
DEFAULT_EXAMPLE_SAT_IMG_2 = "/research/nfs_yilmaz_15/yunus/data/VIGOR/SanFrancisco/satellite/satellite_37.752731622631885_-122.43769208587767.png" # Correct match

def check_image_path(path, label="Image"):
    """ Checks if path exists, prints warning if not. """
    if not path: # Handle None or empty string
        print(f"Warning: No path provided for {label}.")
        return None
    if not os.path.exists(path):
        print(f"Warning: {label} path does not exist: {path}")
    return path

def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM Reranking Test")
    parser.add_argument("--vlm", type=str, default="qwen", choices=["qwen", "llava"], help="VLM model type.")
    parser.add_argument("--model_name", type=str, default=None, help="Override default VLM model name.")
    parser.add_argument("--inference_backend", type=str, default="hf", choices=["hf", "vllm"], help="Inference backend.")

    # --- Strategy Selection ---
    parser.add_argument("--ranking_strategy", type=str, default="pointwise", choices=["pointwise", "pairwise"], help="Ranking strategy.") # Add 'listwise' later

    # --- Image Inputs ---
    parser.add_argument("--query_img", type=str, default=DEFAULT_EXAMPLE_QUERY_IMG, help="Path to the query (ground-level) image.")
    parser.add_argument("--candidate_img_1", type=str, default=DEFAULT_EXAMPLE_SAT_IMG_1, help="Path to the first candidate (satellite) image.")
    parser.add_argument("--candidate_img_2", type=str, default=DEFAULT_EXAMPLE_SAT_IMG_2, help="Path to the second candidate image (for pairwise).")
    # Add more candidate args for listwise later if needed

    # --- Prompt & Generation ---
    parser.add_argument("--prompt_mode", type=str, default="basic", choices=["basic", "reasoning"], help="Prompt detail mode (basic score/pref or with reasoning).")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens for generation.")

    # --- Backend/Hardware ---
    parser.add_argument("--use_quantization", action='store_true', help="Enable quantization (backend-specific method).")
    parser.add_argument("--device", type=str, default="auto", help="Device for HF backend ('auto', 'cuda', 'cpu').")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM.")

    return parser.parse_args()

def main():
    args = parse_args()

    # --- Validate and Load Images based on Strategy ---
    query_img_path = check_image_path(args.query_img, "Query Image")
    candidate_img_1_path = check_image_path(args.candidate_img_1, "Candidate Image 1")
    candidate_img_2_path = None
    images_to_load = [query_img_path, candidate_img_1_path]

    if args.ranking_strategy == 'pairwise':
        print("Pairwise strategy selected. Loading second candidate image.")
        candidate_img_2_path = check_image_path(args.candidate_img_2, "Candidate Image 2")
        if not candidate_img_2_path:
            print("Error: Pairwise strategy requires --candidate_img_2 argument.")
            return
        images_to_load.append(candidate_img_2_path)
    elif args.ranking_strategy == 'listwise':
        # TODO: Add logic for loading multiple candidates for listwise
        print("Error: Listwise strategy not yet implemented.")
        return

    # Ensure all necessary paths exist before proceeding
    if not all(images_to_load):
         print("Error: One or more required image paths are missing or invalid.")
         return

    # --- Select VLM Class ---
    vlm_class = None
    default_model = None
    if args.vlm == "qwen":
        vlm_class = QwenVLM
        default_model = "Qwen/Qwen2.5-VL-7B-Instruct"
    elif args.vlm == "llava":
        vlm_class = LlavaVLM
        default_model = "llava-hf/llava-1.5-7b-hf"
    else: # Should be caught by argparse choices, but good practice
        print(f"Error: VLM type '{args.vlm}' not supported.")
        return

    print(f"Using VLM: {args.vlm}, Strategy: {args.ranking_strategy}, Backend: {args.inference_backend}")

    model_name_to_load = args.model_name if args.model_name else default_model

    # --- Initialize VLM ---
    try:
        print(f"Attempting to load VLM: {model_name_to_load}...")
        vlm_instance = vlm_class(
            model_name=model_name_to_load,
            device=args.device,
            use_quantization=args.use_quantization,
            inference_backend=args.inference_backend,
            tensor_parallel_size=args.tensor_parallel_size,
            # Pass max expected images to vLLM engine args if needed
            # max_images_per_prompt=len(images_to_load)
        )
    except Exception as e:
        print(f"Failed to initialize VLM: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Load PIL Images ---
    images_to_pass = []
    try:
        print("Loading images...")
        for i, img_path in enumerate(images_to_load):
             label = f"Image {i}"
             if i == 0: label = "Query"
             elif i == 1: label = "Candidate 1"
             elif i == 2: label = "Candidate 2"
             print(f" - Loading {label} from: {img_path}")
             images_to_pass.append(vlm_instance.load_image(img_path))
        print(f"Loaded {len(images_to_pass)} images.")
    except Exception as e:
        print(f"Failed to load one or more images: {e}")
        return

    # --- Build Prompt ---
    try:
        # Use the helper function to get the correct builder
        prompt_builder = get_prompt_builder(args.vlm, args.ranking_strategy)
        print(f"Building {args.ranking_strategy} prompt (mode: {args.prompt_mode}) for {args.vlm}...")
        prompt_conversation = prompt_builder(mode=args.prompt_mode)
    except ValueError as e: # Catch errors from get_prompt_builder or builder itself
        print(f"Error building prompt: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while building the prompt: {e}")
        return

    # --- Generate Response ---
    print(f"Generating response using {args.inference_backend} backend...")
    start_time = time.time()
    output_text = "[Generation Error]" # Default error text
    try:
        output_text = vlm_instance.generate(
             conversation=prompt_conversation,
             image_inputs=images_to_pass,
             max_new_tokens=args.max_new_tokens
        )
    except NotImplementedError as e:
         print(f"Generation Error: {e}")
    except RuntimeError as e:
         print(f"Runtime error during generation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during generation: {e}")
        import traceback
        traceback.print_exc()

    end_time = time.time()
    print("-" * 30)
    print(f"Generation took: {end_time - start_time:.2f} seconds")
    print("Model Output:")
    print(output_text)
    print("-" * 30)

    # --- Try parsing the output ---
    try:
        # Clean potential markdown
        if output_text.startswith("```json"):
            output_text = output_text.splitlines()[1].strip()
            if output_text.endswith("```"):
                 output_text = output_text[:-3].strip()

        parsed_output = json.loads(output_text)
        print("Parsed JSON output:")
        print(parsed_output)

        # Check for expected keys based on strategy
        if args.ranking_strategy == 'pointwise':
            if 'score' in parsed_output:
                 print(f"Extracted Score: {parsed_output['score']}")
            if 'reasoning' in parsed_output:
                 print(f"Extracted Reasoning: {parsed_output['reasoning']}")
        elif args.ranking_strategy == 'pairwise':
             if 'preference' in parsed_output:
                  print(f"Extracted Preference (1 or 2): {parsed_output['preference']}")
             if 'reasoning' in parsed_output:
                  print(f"Extracted Reasoning: {parsed_output['reasoning']}")
        # Add parsing for listwise later

    except json.JSONDecodeError:
        print("Warning: Output could not be parsed as JSON.")
    except Exception as e:
        print(f"Error parsing JSON output: {e}")


if __name__ == "__main__":
    main()