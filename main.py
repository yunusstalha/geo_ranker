import argparse
import torch
from PIL import Image
import json # To parse potential JSON output
import time

# Import VLM classes and prompt builders
# Import LlavaVLM only when needed to avoid dependency errors if not used
# from models import QwenVLM, LlavaVLM # Adjust based on which you test
from models import QwenVLM

from prompts import prompt_templates

# --- Example Image Paths (Ensure these are correct on your system) ---
EXAMPLE_QUERY_IMG = "/research/nfs_yilmaz_15/yunus/data/VIGOR/SanFrancisco/panorama/ScGwgVhnS9mLGl47wRnicQ,37.752695,-122.437849,.jpg"
EXAMPLE_SAT_IMG_CORRECT = "/research/nfs_yilmaz_15/yunus/data/VIGOR/SanFrancisco/satellite/satellite_37.752731622631885_-122.43810708310743.png"
EXAMPLE_SAT_IMG_WRONG = "/research/nfs_yilmaz_15/yunus/data/VIGOR/SanFrancisco/satellite/satellite_37.752731622631885_-122.43769208587767.png"
# --- ---

def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM Reranking Test")
    parser.add_argument("--vlm", type=str, default="qwen", choices=["qwen", "llava"], help="VLM model type to use.")
    parser.add_argument("--model_name", type=str, default=None, help="Override default model name (e.g., Qwen/Qwen2.5-VL-7B-Instruct or llava-hf/llava-1.5-7b-hf)")
    parser.add_argument("--query_img", type=str, default=EXAMPLE_QUERY_IMG, help="Path to the query image.")
    parser.add_argument("--candidate_img", type=str, default=EXAMPLE_SAT_IMG_CORRECT, help="Path to the candidate satellite image.")
    parser.add_argument("--prompt_mode", type=str, default="basic", choices=["basic", "reasoning"], help="Pointwise prompt mode.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens for generation.")
    parser.add_argument("--use_quantization", action='store_true', help="Enable 4-bit quantization using BitsAndBytes.") # Keep note this uses BnB now
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('auto', 'cuda', 'cpu')")

    return parser.parse_args()

def main():
    args = parse_args()

    # --- Select VLM and Default Model Name ---
    vlm_class = None
    default_model = None
    prompt_builder = None

    if args.vlm == "qwen":
        vlm_class = QwenVLM
        default_model = "Qwen/Qwen2.5-VL-7B-Instruct"
        prompt_builder = prompt_templates.build_pointwise_qwen
        print("Using Qwen VLM.")
    elif args.vlm == "llava":
        try:
            # Dynamically import LlavaVLM
            from models import LlavaVLM
            # Update __init__.py __all__ list if using Llava permanently
            # Example: __all__ = ['BaseVLM', 'QwenVLM', 'LlavaVLM']
            vlm_class = LlavaVLM
            default_model = "llava-hf/llava-1.5-7b-hf" # Example Llava model
            prompt_builder = prompt_templates.build_pointwise_llava
            print("Using LLaVA VLM.")
        except ImportError:
             print("Error: Could not import LlavaVLM. Make sure it's implemented, dependencies (llava-hf) are installed, and it's included in models/__init__.py if needed.")
             return
        except NameError:
             print("Error: LlavaVLM class not found. Check implementation and imports.")
             return
    else:
        print(f"Error: VLM type '{args.vlm}' not supported.")
        return

    model_name_to_load = args.model_name if args.model_name else default_model

    # --- Initialize VLM ---
    try:
        print(f"Attempting to load VLM: {model_name_to_load}")
        vlm_instance = vlm_class(
            model_name=model_name_to_load,
            device=args.device,
            use_quantization=args.use_quantization
        )
    except Exception as e:
        print(f"Failed to initialize VLM: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return

    # --- Load Images ---
    print(f"Loading query image: {args.query_img}")
    print(f"Loading candidate image: {args.candidate_img}")
    try:
        query_image = vlm_instance.load_image(args.query_img)
        candidate_image = vlm_instance.load_image(args.candidate_img)
        images_to_pass = [query_image, candidate_image]
    except Exception as e:
        print(f"Failed to load images: {e}")
        return

    # --- Build Prompt ---
    print(f"Building pointwise prompt (mode: {args.prompt_mode}) for {args.vlm}...")
    try:
        # Get the prompt structure (conversation list)
        prompt_conversation = prompt_builder(mode=args.prompt_mode)
    except ValueError as e:
        print(f"Error building prompt: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while building the prompt: {e}")
        return

    # --- Generate Response ---
    print("Generating response...")
    start_time = time.time()
    try:
        # Pass the conversation list structure
        output_text = vlm_instance.generate(
             conversation=prompt_conversation,
             image_inputs=images_to_pass,
             max_new_tokens=args.max_new_tokens
        )
    except RuntimeError as e:
         print(f"Runtime error during generation: {e}")
         return
    except Exception as e:
        print(f"An unexpected error occurred during generation: {e}")
        import traceback
        traceback.print_exc()
        return

    end_time = time.time()
    print("-" * 30)
    print(f"Generation took: {end_time - start_time:.2f} seconds")
    print("Model Output:")
    print(output_text)
    print("-" * 30)

    # --- Try parsing the output (optional) ---
    try:
        # Simple parsing assuming the JSON format is respected
        # Remove potential markdown backticks ```json ... ```
        if output_text.startswith("```json"):
            output_text = output_text.splitlines()[1].strip()
            if output_text.endswith("```"):
                 output_text = output_text[:-3].strip()

        parsed_output = json.loads(output_text)
        print("Parsed JSON output:")
        print(parsed_output)
        if 'score' in parsed_output:
             print(f"Extracted Score: {parsed_output['score']}")
        if 'reasoning' in parsed_output:
             print(f"Extracted Reasoning: {parsed_output['reasoning']}")
    except json.JSONDecodeError:
        print("Output is not valid JSON.")
    except Exception as e:
        print(f"Error parsing JSON output: {e}")


if __name__ == "__main__":
    main()