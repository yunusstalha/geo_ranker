# prompts/prompt_templates.py
from typing import List, Dict, Union # Added typing

"""
Prompt formats for different strategies and modes.
"""

# --- Pointwise Qwen ---
def build_pointwise_qwen(mode='basic', reasoning_text=None):
    """ Builds pointwise prompts for Qwen-VL. """
    content_base = [
        {"type": "text", "text": "Here is the ground-level panorama query image:"},
        {"type": "image"}, # Placeholder for query image (index 0)
        {"type": "text", "text": "Here is a candidate satellite image:"},
        {"type": "image"}, # Placeholder for candidate image (index 1)
    ]

    if mode == 'basic':
        content_final = content_base + [
            {"type": "text", "text": "Evaluate if the satellite image corresponds to the location shown in the ground-level panorama."},
            {"type": "text", "text": "Provide a confidence score between 0 (no match) and 100 (perfect match)."},
            {"type": "text", "text": "Respond ONLY with a JSON object containing the score, like this: {\"score\": <score_value>}"}
        ]
    elif mode == 'reasoning': # Keep single-pass reasoning for pointwise
        content_final = content_base + [
            {"type": "text", "text": "Evaluate if the satellite image corresponds to the location shown in the ground-level panorama."},
            {"type": "text", "text": "First, provide a brief step-by-step reasoning comparing key visual features (e.g., road layout, building shapes, landmarks)."},
            {"type": "text", "text": "Then, provide a confidence score between 0 (no match) and 100 (perfect match)."},
            {"type": "text", "text": "Respond ONLY with a JSON object containing the reasoning and score, like this: {\"reasoning\": \"<your_reasoning>\", \"score\": <score_value>}"}
        ]
    elif mode == 'yesno':
        content_final = content_base + [
            {"type": "text", "text": "Does the satellite image accurately match the location shown in the ground-level panorama?"},
            {"type": "text", "text": "Answer ONLY with the single word 'Yes' or 'No'."}
        ]
    elif mode == 'likert':
        content_final = content_base + [
            {"type": "text", "text": "On a scale of 1 (no match) to 5 (perfect match), how well does the satellite image match the location in the ground-level panorama?"},
            {"type": "text", "text": "Respond ONLY with a single digit (1, 2, 3, 4, or 5)."}
        ]
    elif mode == 'reasoning_only': # For two-pass pointwise
         content_final = content_base + [
            {"type": "text", "text": "Evaluate if the satellite image corresponds to the location shown in the ground-level panorama."},
            {"type": "text", "text": "Provide a detailed step-by-step reasoning comparing key visual features (e.g., road layout, building shapes, relative positions, landmarks). Explain your conclusion."},
            {"type": "text", "text": "Focus on providing only the reasoning text."}
        ]
    elif mode == 'score_from_reasoning': # For two-pass pointwise
        if reasoning_text is None:
            raise ValueError("Reasoning text must be provided for 'score_from_reasoning' mode.")
        content_final = content_base + [
            {"type": "text", "text": "Based on the following reasoning about the match between the two images:"},
            {"type": "text", "text": f"Reasoning: \"{reasoning_text}\""},
            {"type": "text", "text": "Provide a final confidence score between 0 (no match) and 100 (perfect match)."},
            {"type": "text", "text": "Respond ONLY with a JSON object containing the score, like this: {\"score\": <score_value>}"}
        ]
    elif mode == 'yesno_from_reasoning': # For two-pass pointwise
        if reasoning_text is None:
            raise ValueError("Reasoning text must be provided for 'yesno_from_reasoning' mode.")
        content_final = content_base + [
            {"type": "text", "text": "Based on the following reasoning about the match between the two images:"},
            {"type": "text", "text": f"Reasoning: \"{reasoning_text}\""},
            {"type": "text", "text": "Concisely, does the satellite image match the ground-level panorama?"},
            {"type": "text", "text": "Answer ONLY with the single word 'Yes' or 'No'."}
        ]
    else:
        raise ValueError(f"Unknown pointwise mode for Qwen: {mode}")

    conversation = [{"role": "user", "content": content_final}]
    return conversation


# --- Pairwise Qwen ---
def build_pairwise_qwen(mode='basic', reasoning_text=None):
    """ Builds pairwise prompts for Qwen-VL, including two-pass modes (no confidence). """
    task_description = ""
    response_format = ""
    # --- Common Introduction ---
    intro = [
        {"type": "text", "text": "You are an expert geospatial analyst. Your task is to determine the best satellite image match for a ground-level query panorama."},
        {"type": "text", "text": "Here is the ground-level panorama query image:"},
        {"type": "image"}, # Index 0 (Query)
        {"type": "text", "text": "Here is Satellite Image 1 (Candidate 1):"},
        {"type": "image"}, # Index 1 (Cand 1)
        {"type": "text", "text": "Here is Satellite Image 2 (Candidate 2):"},
        {"type": "image"}, # Index 2 (Cand 2)
    ]

    if mode == 'basic':
        task_description = (
            "Carefully analyze the ground-level query image and compare it to both candidate satellite images. "
            "Focus on key cross-view features like road network, building arrangement, and landmarks.\n\n"
            "Based on this comparison, which satellite image (1 or 2) provides the **better geospatial match**?"
        )
        response_format = "Respond ONLY with a JSON object indicating the preferred image number (1 or 2), like this: {\"preference\": <1_or_2>}"
        content = intro + [
            {"type": "text", "text": task_description},
            {"type": "text", "text": response_format}
        ]

    elif mode == 'reasoning_only':
        task_description = (
            "Carefully analyze the ground-level query image and compare it to both candidate satellite images (Satellite Image 1, Satellite Image 2).\n\n"
            "Provide a detailed step-by-step reasoning comparing these specific aspects:\n"
            "1.  **Road Network:** Alignment of road layouts (intersections, curves, widths).\n"
            "2.  **Building Arrangement:** Similarity of building footprints, shapes, and relative positioning.\n"
            "3.  **Landmarks/Open Spaces:** Matching or conflicting landmarks or large open areas.\n\n"
            "Focus ONLY on generating the comparative reasoning text. Do not state a preference yet."
        )
        # No specific response format needed, just expect text.
        content = intro + [{"type": "text", "text": task_description}]

    elif mode == 'preference_from_reasoning': 
        if reasoning_text is None:
            raise ValueError("Reasoning text must be provided for 'preference_from_reasoning' mode.")
        task_description = (
            "Based on the images provided above and the following detailed reasoning derived from comparing them:\n\n"
            f"--- Reasoning Start ---\n{reasoning_text}\n--- Reasoning End ---\n\n"
            "Which satellite image (1 or 2) shows **stronger correspondence** and is the **superior geospatial match** to the query panorama?"
        )
        response_format = ( 
            "Respond ONLY with a JSON object containing the preferred image number (1 or 2), structured like this:\n"
            "{\"preference\": <1_or_2>}"
        )
        content = intro + [
            {"type": "text", "text": task_description},
            {"type": "text", "text": response_format}
        ]
    else:
        raise ValueError(f"Unknown or unsupported pairwise mode for Qwen: {mode}. Use 'basic', 'reasoning_only', or 'preference_from_reasoning'.")

    conversation = [{"role": "user", "content": content}]
    return conversation

# --- Listwise Qwen ---
def build_listwise_qwen(num_candidates: int):
    """ Builds listwise prompts for Qwen-VL. """
    if num_candidates < 2:
        raise ValueError("Listwise requires at least 2 candidates.")

    content = [
        {"type": "text", "text": "Here is the ground-level panorama query image:"},
        {"type": "image"}, # Index 0 (Query)
    ]
    for i in range(num_candidates):
        content.extend([
            {"type": "text", "text": f"Candidate Satellite Image {i+1}:"},
            {"type": "image"}, # Index 1, 2, ... num_candidates
        ])
    content.extend([
        {"type": "text", "text": f"Rank the {num_candidates} candidate satellite images based on how well they match the location shown in the query image."},
        {"type": "text", "text": f"List the image numbers (1 to {num_candidates}) from best match to worst match."},
        {"type": "text", "text": f"Respond ONLY with a JSON object containing the ranked list, like this: {{\"ranking\": [<best_image_number>, <second_best_number>, ..., <worst_image_number>]}}"}
        # Example for 3 candidates: {"ranking": [2, 1, 3]}
    ])
    conversation = [{"role": "user", "content": content}]
    return conversation


# --- Pointwise LLaVA ---
def build_pointwise_llava(mode='basic', reasoning_text=None):
    """ Builds pointwise prompts for Llava-1.5. """
    content_base = [
        {"type": "text", "text": "Here is the ground-level panorama query image:"},
        {"type": "image"}, # Index 0
        {"type": "text", "text": "Here is a candidate satellite image:"},
        {"type": "image"}, # Index 1
    ]

    if mode == 'basic':
        content_final = content_base + [
            {"type": "text", "text": "Evaluate if the satellite image corresponds to the location shown in the ground-level panorama."},
            {"type": "text", "text": "Provide a confidence score between 0 (no match) and 100 (perfect match)."},
            {"type": "text", "text": "Respond ONLY with a JSON object containing the score, like this: {\"score\": <score_value>}"}
        ]
    elif mode == 'reasoning': # Keep single-pass reasoning for pointwise
        content_final = content_base + [
            {"type": "text", "text": "Evaluate if the satellite image corresponds to the location shown in the ground-level panorama."},
            {"type": "text", "text": "First, provide a brief step-by-step reasoning comparing key visual features (e.g., road layout, building shapes, landmarks)."},
            {"type": "text", "text": "Then, provide a confidence score between 0 (no match) and 100 (perfect match)."},
            {"type": "text", "text": "Respond ONLY with a JSON object containing the reasoning and score, like this: {\"reasoning\": \"<your_reasoning>\", \"score\": <score_value>}"}
        ]
    elif mode == 'yesno':
        content_final = content_base + [
            {"type": "text", "text": "Does the satellite image accurately match the location shown in the ground-level panorama?"},
            {"type": "text", "text": "Answer ONLY with the single word 'Yes' or 'No'."}
        ]
    elif mode == 'likert':
        content_final = content_base + [
            {"type": "text", "text": "On a scale of 1 (no match) to 5 (perfect match), how well does the satellite image match the location in the ground-level panorama?"},
            {"type": "text", "text": "Respond ONLY with a single digit (1, 2, 3, 4, or 5)."}
        ]
    elif mode == 'reasoning_only': # For two-pass pointwise
         content_final = content_base + [
            {"type": "text", "text": "Evaluate if the satellite image corresponds to the location shown in the ground-level panorama."},
            {"type": "text", "text": "Provide a detailed step-by-step reasoning comparing key visual features (e.g., road layout, building shapes, relative positions, landmarks). Explain your conclusion."},
            {"type": "text", "text": "Focus on providing only the reasoning text."}
        ]
    elif mode == 'score_from_reasoning': # For two-pass pointwise
        if reasoning_text is None:
            raise ValueError("Reasoning text must be provided for 'score_from_reasoning' mode.")
        content_final = content_base + [
            {"type": "text", "text": "Based on the following reasoning about the match between the two images:"},
            {"type": "text", "text": f"Reasoning:\n{reasoning_text}\n---"},
            {"type": "text", "text": "Provide a final confidence score between 0 (no match) and 100 (perfect match)."},
            {"type": "text", "text": "Respond ONLY with a JSON object containing the score, like this: {\"score\": <score_value>}"}
        ]
    elif mode == 'yesno_from_reasoning': # For two-pass pointwise
        if reasoning_text is None:
            raise ValueError("Reasoning text must be provided for 'yesno_from_reasoning' mode.")
        content_final = content_base + [
            {"type": "text", "text": "Based on the following reasoning about the match between the two images:"},
            {"type": "text", "text": f"Reasoning:\n{reasoning_text}\n---"},
            {"type": "text", "text": "Concisely, does the satellite image match the ground-level panorama?"},
            {"type": "text", "text": "Answer ONLY with the single word 'Yes' or 'No'."}
        ]
    else:
        raise ValueError(f"Unknown pointwise mode for LLaVA: {mode}")

    conversation = [{"role": "user", "content": content_final}]
    return conversation

# --- Pairwise LLaVA ---
def build_pairwise_llava(mode='basic', reasoning_text=None):
    """ Builds pairwise prompts for Llava-1.5, including two-pass modes (no confidence). """
    task_description = ""
    response_format = ""
    # --- Common Introduction (LLaVA style) ---
    content = [
        {"type": "text", "text": "You are an expert geospatial analyst. Your task is to determine the best satellite image match for a ground-level query panorama."},
        {"type": "text", "text": "Query Image:"},
        {"type": "image"}, # Index 0 (Query)
        {"type": "text", "text": "Satellite Image 1 (Candidate 1):"},
        {"type": "image"}, # Index 1 (Cand 1)
        {"type": "text", "text": "Satellite Image 2 (Candidate 2):"},
        {"type": "image"}, # Index 2 (Cand 2)
    ]

    if mode == 'basic':
        task_description = (
            "Carefully analyze the ground-level query image and compare it to both candidate satellite images. "
            "Focus on key cross-view features like road network, building arrangement, and landmarks.\n\n"
            "Based on this comparison, which satellite image (1 or 2) provides the **better geospatial match**?"
        )
        response_format = "Respond ONLY with a JSON object indicating the preferred image number (1 or 2), like this: {\"preference\": <1_or_2>}"
        content.extend([
            {"type": "text", "text": task_description},
            {"type": "text", "text": response_format}
        ])

    elif mode == 'reasoning_only': 
        task_description = (
            "Carefully analyze the ground-level query image and compare it to both candidate satellite images (Satellite Image 1, Satellite Image 2).\n\n"
            "Provide a detailed step-by-step reasoning comparing these specific aspects:\n"
            "1.  **Road Network:** Alignment of road layouts (intersections, curves, widths).\n"
            "2.  **Building Arrangement:** Similarity of building footprints, shapes, and relative positioning.\n"
            "3.  **Landmarks/Open Spaces:** Matching or conflicting landmarks or large open areas.\n\n"
            "Focus ONLY on generating the comparative reasoning text. Do not state a preference yet."
        )
        content.append({"type": "text", "text": task_description})

    elif mode == 'preference_from_reasoning': 
        if reasoning_text is None:
            raise ValueError("Reasoning text must be provided for 'preference_from_reasoning' mode.")
        task_description = (
            "Based on the images provided above and the following detailed reasoning derived from comparing them:\n\n"
            f"--- Reasoning Start ---\n{reasoning_text}\n--- Reasoning End ---\n\n"
            "Which satellite image (1 or 2) shows **stronger correspondence** and is the **superior geospatial match** to the query panorama?"
        )
        response_format = (
            "Respond ONLY with a JSON object containing the preferred image number (1 or 2), structured like this:\n"
            "{\"preference\": <1_or_2>}"
        )
        content.extend([
            {"type": "text", "text": task_description},
            {"type": "text", "text": response_format}
        ])
    else:
        raise ValueError(f"Unknown or unsupported pairwise mode for LLaVA: {mode}. Use 'basic', 'reasoning_only', or 'preference_from_reasoning'.")

    conversation = [{"role": "user", "content": content}]
    return conversation

# --- Listwise LLaVA ---
def build_listwise_llava(num_candidates: int):
    """ Builds listwise prompts for Llava-1.5. """
    if num_candidates < 2:
        raise ValueError("Listwise requires at least 2 candidates.")

    content = [
        {"type": "text", "text": "Here is the ground-level panorama query image:"},
        {"type": "image"}, # Index 0 (Query)
    ]
    for i in range(num_candidates):
        content.extend([
            {"type": "text", "text": f"Candidate Satellite Image {i+1}:"},
            {"type": "image"}, # Index 1, 2, ... num_candidates
        ])
    content.extend([
        {"type": "text", "text": f"Rank the {num_candidates} candidate satellite images based on how well they match the location shown in the query image."},
        {"type": "text", "text": f"List the image numbers (1 to {num_candidates}) from best match to worst match."},
        {"type": "text", "text": f"Respond ONLY with a JSON object containing the ranked list, like this: {{\"ranking\": [<best_image_number>, <second_best_number>, ..., <worst_image_number>]}}"}
        # Example for 3 candidates: {"ranking": [2, 1, 3]}
    ])
    conversation = [{"role": "user", "content": content}]
    return conversation


# --- Get Prompt Builder ---
def get_prompt_builder(vlm_type: str, strategy: str):
    """
    Returns the correct base prompt builder function based on VLM and strategy.
    The specific 'mode' and 'reasoning_text' are passed by the caller.
    """
    # Map VLM type and strategy to the base builder function
    base_builder_map = {
        ('qwen', 'pointwise'): build_pointwise_qwen,
        ('qwen', 'pairwise'): build_pairwise_qwen,
        ('qwen', 'listwise'): build_listwise_qwen, # Assuming you might add later
        ('llava', 'pointwise'): build_pointwise_llava,
        ('llava', 'pairwise'): build_pairwise_llava,
        ('llava', 'listwise'): build_listwise_llava, # Assuming you might add later
    }

    builder_func = base_builder_map.get((vlm_type.lower(), strategy.lower()))

    if builder_func is None:
        # Adjusted the error message slightly
        raise ValueError(f"No prompt builder function found for VLM='{vlm_type}', Strategy='{strategy}'")

    return builder_func