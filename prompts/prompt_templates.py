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
    elif mode == 'reasoning':
        content_final = content_base + [
            {"type": "text", "text": "Evaluate if the satellite image corresponds to the location shown in the ground-level panorama."},
            {"type": "text", "text": "First, provide a brief step-by-step reasoning comparing key visual features (e.g., road layout, building shapes, landmarks)."},
            {"type": "text", "text": "Then, provide a confidence score between 0 (no match) and 100 (perfect match)."},
            {"type": "text", "text": "Respond ONLY with a JSON object containing the reasoning and score, like this: {\"reasoning\": \"<your_reasoning>\", \"score\": <score_value>}"}
        ]
    elif mode == 'yesno': # New mode
        content_final = content_base + [
            {"type": "text", "text": "Does the satellite image accurately match the location shown in the ground-level panorama?"},
            {"type": "text", "text": "Answer ONLY with the single word 'Yes' or 'No'."} # Instruction for logit scoring
        ]
    elif mode == 'likert': # New mode
        content_final = content_base + [
            {"type": "text", "text": "On a scale of 1 (no match) to 5 (perfect match), how well does the satellite image match the location in the ground-level panorama?"},
            {"type": "text", "text": "Respond ONLY with a single digit (1, 2, 3, 4, or 5)."} # Instruction for logit scoring
        ]
    elif mode == 'reasoning_only': # New mode for two-pass
         content_final = content_base + [
            {"type": "text", "text": "Evaluate if the satellite image corresponds to the location shown in the ground-level panorama."},
            {"type": "text", "text": "Provide a detailed step-by-step reasoning comparing key visual features (e.g., road layout, building shapes, relative positions, landmarks). Explain your conclusion."},
            {"type": "text", "text": "Focus on providing only the reasoning text."}
        ]
    elif mode == 'score_from_reasoning': # New mode for two-pass (JSON)
        if reasoning_text is None:
            raise ValueError("Reasoning text must be provided for 'score_from_reasoning' mode.")
        content_final = content_base + [
            {"type": "text", "text": "Based on the following reasoning about the match between the two images:"},
            {"type": "text", "text": f"Reasoning: \"{reasoning_text}\""},
            {"type": "text", "text": "Provide a final confidence score between 0 (no match) and 100 (perfect match)."},
            {"type": "text", "text": "Respond ONLY with a JSON object containing the score, like this: {\"score\": <score_value>}"}
        ]
    elif mode == 'yesno_from_reasoning': # New mode for two-pass (Yes/No)
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
def build_pairwise_qwen(mode='basic'):
    """ Builds pairwise prompts for Qwen-VL. """
    # (Existing pairwise logic - no changes needed for now)
    task_description = ""
    response_format = ""
    if mode == 'basic':
        task_description = "Compare Satellite Image 1 and Satellite Image 2 to the ground-level query image. Which satellite image (1 or 2) shows the same location as the query image?"
        response_format = "Respond ONLY with a JSON object indicating the preferred image number (1 or 2), like this: {\"preference\": <1_or_2>}"
    elif mode == 'reasoning':
        task_description = "Compare Satellite Image 1 and Satellite Image 2 to the ground-level query image. Which satellite image (1 or 2) shows the same location as the query image? Provide a brief step-by-step reasoning for your choice based on visual features."
        response_format = "Respond ONLY with a JSON object containing the preference (1 or 2) and reasoning, like this: {\"preference\": <1_or_2>, \"reasoning\": \"<your_reasoning>\"}"
    else:
        raise ValueError(f"Unknown pairwise mode for Qwen: {mode}")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the ground-level panorama query image:"},
                {"type": "image"}, # Index 0
                {"type": "text", "text": "Here is Satellite Image 1 (Candidate 1):"},
                {"type": "image"}, # Index 1
                {"type": "text", "text": "Here is Satellite Image 2 (Candidate 2):"},
                {"type": "image"}, # Index 2
                {"type": "text", "text": task_description},
                {"type": "text", "text": response_format}
            ]
        }
    ]
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
    elif mode == 'reasoning':
        content_final = content_base + [
            {"type": "text", "text": "Evaluate if the satellite image corresponds to the location shown in the ground-level panorama."},
            {"type": "text", "text": "First, provide a brief step-by-step reasoning comparing key visual features (e.g., road layout, building shapes, landmarks)."},
            {"type": "text", "text": "Then, provide a confidence score between 0 (no match) and 100 (perfect match)."},
            {"type": "text", "text": "Respond ONLY with a JSON object containing the reasoning and score, like this: {\"reasoning\": \"<your_reasoning>\", \"score\": <score_value>}"}
        ]
    elif mode == 'yesno': # New mode
        content_final = content_base + [
            {"type": "text", "text": "Does the satellite image accurately match the location shown in the ground-level panorama?"},
            {"type": "text", "text": "Answer ONLY with the single word 'Yes' or 'No'."} # Instruction for logit scoring
        ]
    elif mode == 'likert': # New mode
        content_final = content_base + [
            {"type": "text", "text": "On a scale of 1 (no match) to 5 (perfect match), how well does the satellite image match the location in the ground-level panorama?"},
            {"type": "text", "text": "Respond ONLY with a single digit (1, 2, 3, 4, or 5)."} # Instruction for logit scoring
        ]
    elif mode == 'reasoning_only': # New mode for two-pass
         content_final = content_base + [
            {"type": "text", "text": "Evaluate if the satellite image corresponds to the location shown in the ground-level panorama."},
            {"type": "text", "text": "Provide a detailed step-by-step reasoning comparing key visual features (e.g., road layout, building shapes, relative positions, landmarks). Explain your conclusion."},
            {"type": "text", "text": "Focus on providing only the reasoning text."}
        ]
    elif mode == 'score_from_reasoning': # New mode for two-pass (JSON)
        if reasoning_text is None:
            raise ValueError("Reasoning text must be provided for 'score_from_reasoning' mode.")
        content_final = content_base + [
            {"type": "text", "text": "Based on the following reasoning about the match between the two images:"},
            {"type": "text", "text": f"Reasoning:\n{reasoning_text}\n---"}, # Use reasoning text
            {"type": "text", "text": "Provide a final confidence score between 0 (no match) and 100 (perfect match)."},
            {"type": "text", "text": "Respond ONLY with a JSON object containing the score, like this: {\"score\": <score_value>}"}
        ]
    elif mode == 'yesno_from_reasoning': # New mode for two-pass (Yes/No)
        if reasoning_text is None:
            raise ValueError("Reasoning text must be provided for 'yesno_from_reasoning' mode.")
        content_final = content_base + [
            {"type": "text", "text": "Based on the following reasoning about the match between the two images:"},
            {"type": "text", "text": f"Reasoning:\n{reasoning_text}\n---"}, # Use reasoning text
            {"type": "text", "text": "Concisely, does the satellite image match the ground-level panorama?"},
            {"type": "text", "text": "Answer ONLY with the single word 'Yes' or 'No'."}
        ]
    else:
        raise ValueError(f"Unknown pointwise mode for LLaVA: {mode}")

    conversation = [{"role": "user", "content": content_final}]
    return conversation


# --- Pairwise LLaVA ---
def build_pairwise_llava(mode='basic'):
    """ Builds pairwise prompts for Llava-1.5. """
    # (Existing pairwise logic - no changes needed for now)
    task_description = ""
    response_format = ""
    if mode == 'basic':
        task_description = "Compare Satellite Image 1 and Satellite Image 2 to the ground-level query image. Which satellite image (1 or 2) shows the same location as the query image?"
        response_format = "Respond ONLY with a JSON object indicating the preferred image number (1 or 2), like this: {\"preference\": <1_or_2>}"
    elif mode == 'reasoning':
        task_description = "Compare Satellite Image 1 and Satellite Image 2 to the ground-level query image. Which satellite image (1 or 2) shows the same location as the query image? Provide a brief step-by-step reasoning for your choice based on visual features."
        response_format = "Respond ONLY with a JSON object containing the preference (1 or 2) and reasoning, like this: {\"preference\": <1_or_2>, \"reasoning\": \"<your_reasoning>\"}"
    else:
        raise ValueError(f"Unknown pairwise mode for LLaVA: {mode}")

    content = [
        {"type": "text", "text": "Here is the ground-level panorama query image:"},
        {"type": "image"}, # Index 0
        {"type": "text", "text": "Here is Satellite Image 1 (Candidate 1):"},
        {"type": "image"}, # Index 1
        {"type": "text", "text": "Here is Satellite Image 2 (Candidate 2):"},
        {"type": "image"}, # Index 2
        {"type": "text", "text": task_description},
        {"type": "text", "text": response_format}
    ]
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
def get_prompt_builder(vlm_type: str, strategy: str, mode: str):
    """
    Returns the correct prompt builder function based on VLM, strategy, and mode.
    Passes reasoning_text if the mode requires it.
    """
    # Adjusted map to handle different modes within strategies
    builder_map = {
        ('qwen', 'pointwise', 'basic'): build_pointwise_qwen,
        ('qwen', 'pointwise', 'reasoning'): build_pointwise_qwen,
        ('qwen', 'pointwise', 'yesno'): build_pointwise_qwen,
        ('qwen', 'pointwise', 'likert'): build_pointwise_qwen,
        ('qwen', 'pointwise', 'reasoning_only'): build_pointwise_qwen,
        ('qwen', 'pointwise', 'score_from_reasoning'): build_pointwise_qwen,
        ('qwen', 'pointwise', 'yesno_from_reasoning'): build_pointwise_qwen,
        ('qwen', 'pairwise', 'basic'): build_pairwise_qwen,
        ('qwen', 'pairwise', 'reasoning'): build_pairwise_qwen,
        #('qwen', 'listwise', 'basic'): build_listwise_qwen, # Add when ready

        ('llava', 'pointwise', 'basic'): build_pointwise_llava,
        ('llava', 'pointwise', 'reasoning'): build_pointwise_llava,
        ('llava', 'pointwise', 'yesno'): build_pointwise_llava,
        ('llava', 'pointwise', 'likert'): build_pointwise_llava,
        ('llava', 'pointwise', 'reasoning_only'): build_pointwise_llava,
        ('llava', 'pointwise', 'score_from_reasoning'): build_pointwise_llava,
        ('llava', 'pointwise', 'yesno_from_reasoning'): build_pointwise_llava,
        ('llava', 'pairwise', 'basic'): build_pairwise_llava,
        ('llava', 'pairwise', 'reasoning'): build_pairwise_llava,
        #('llava', 'listwise', 'basic'): build_listwise_llava, # Add when ready
    }

    builder = builder_map.get((vlm_type.lower(), strategy.lower(), mode.lower()))

    if builder is None:
        raise ValueError(f"No prompt builder found for VLM='{vlm_type}', Strategy='{strategy}', Mode='{mode}'")

    # Return the builder function itself. The caller will pass the mode and optional reasoning_text.
    return builder