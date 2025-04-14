# src/reranking/prompts.py
from typing import List

# --- Common Placeholders ---
# {query_desc}: A textual description of the query (e.g., "this street view panorama")
# {image_placeholder}: Token indicating where an image should be inserted (e.g., "<image>")
# {candidate_indices}: e.g., "Candidate 1", "Candidate 2"
# {candidate_descriptions}: e.g., "Satellite Image 1", "Satellite Image 2"

# --- Pointwise Prompts ---
POINTWISE_PROMPTS = {
    "basic": [ # Requires likelihood scoring
        "{query_desc}: {image_placeholder}. ",
        "How relevant is this satellite image: {image_placeholder} to the query?"
        # Expected targets for likelihood: "Relevant", "Not Relevant" or similar
    ],
    "reasoning": [
        "{query_desc}: {image_placeholder}. ",
        "Consider the satellite image: {image_placeholder}. ",
        "Explain step-by-step why this satellite image does or does not match the query image, considering features like buildings, roads, and layout. Based on your reasoning, is it a strong match?"
    ],
    "score": [
        "{query_desc}: {image_placeholder}. ",
        "Consider the satellite image: {image_placeholder}. ",
        "On a scale of 0 to 10, how well does this satellite image match the query image based on visual features (buildings, roads, layout, vegetation)? Provide only the numerical score."
    ],
     "score_reasoning": [ # Combines scoring and reasoning
        "{query_desc}: {image_placeholder}. ",
        "Consider the satellite image: {image_placeholder}. ",
        "Explain step-by-step why this satellite image does or does not match the query image, considering features like buildings, roads, and layout. Based on your reasoning, assign a matching score from 0 (no match) to 10 (perfect match). Provide the reasoning first, then the score on a new line like 'Score: X'."
    ]
}

# --- Pairwise Prompts ---
PAIRWISE_PROMPTS = {
    "basic": [
        "{query_desc}: {image_placeholder}. ",
        "Now consider two candidate satellite images. Candidate A: {image_placeholder}. Candidate B: {image_placeholder}. ",
        "Which satellite image (A or B) is a better match for the query image? Answer with only 'A' or 'B'."
    ],
    "reasoning": [
        "{query_desc}: {image_placeholder}. ",
        "Now consider two candidate satellite images. Candidate A: {image_placeholder}. Candidate B: {image_placeholder}. ",
        "Compare Candidate A and Candidate B to the query image. Explain the key visual similarities and differences for each candidate relative to the query. Based on your comparison, which satellite image (A or B) is the better match? Provide the reasoning first, then the final choice on a new line like 'Choice: A' or 'Choice: B'."
    ],
     # Score-based pairwise might involve asking for scores for both and comparing,
     # or asking for a relative preference score. Let's stick to choice for now.
}

# --- Listwise Prompts ---
# Note: Handling many images might exceed context limits. Placeholders assume indices 1 to N.
# The VLM needs to be instructed *very clearly* on the output format.
LISTWISE_PROMPTS = {
    "basic": [
        "{query_desc}: {image_placeholder}. ",
        "Here are {num_candidates} candidate satellite images: ",
        # Loop this part in the strategy code: "Candidate {i+1}: {image_placeholder}. "
        "Rank these {num_candidates} satellite images from best match (most similar) to worst match (least similar) for the query image. Output the ranking as a comma-separated list of candidate indices (e.g., '3, 1, 4, 2' if Candidate 3 is best, followed by 1, etc.). Provide only the ranked list."
    ],
    "reasoning": [
        "{query_desc}: {image_placeholder}. ",
        "Here are {num_candidates} candidate satellite images: ",
        # Loop this part: "Candidate {i+1}: {image_placeholder}. "
        "For each candidate satellite image (1 to {num_candidates}), briefly explain its visual similarity or dissimilarity to the query image. Then, rank all {num_candidates} candidates from best match (most similar) to worst match (least similar). Output the reasoning first, then the final ranking on a new line as a comma-separated list of indices (e.g., 'Ranking: 3, 1, 4, 2')."
     ],
     "score": [
        "{query_desc}: {image_placeholder}. ",
        "Here are {num_candidates} candidate satellite images: ",
        # Loop this part: "Candidate {i+1}: {image_placeholder}. "
        "Assign a matching score from 0 (worst) to 10 (best) for each candidate satellite image (1 to {num_candidates}) compared to the query image. Output the scores as a comma-separated list, corresponding to the candidate indices (e.g., 'Candidate 1 score, Candidate 2 score, ...'). Provide only the list of scores."
     ]
}


def get_prompt_template(strategy: str, prompt_type: str) -> List[str]:
    """
    Retrieves the appropriate prompt template list.

    Args:
        strategy (str): 'pointwise', 'pairwise', or 'listwise'.
        prompt_type (str): 'basic', 'reasoning', 'score', etc.

    Returns:
        List[str]: The list of text parts for the prompt template.

    Raises:
        ValueError: If the strategy or type is invalid.
    """
    templates = {}
    if strategy == 'pointwise':
        templates = POINTWISE_PROMPTS
    elif strategy == 'pairwise':
        templates = PAIRWISE_PROMPTS
    elif strategy == 'listwise':
        templates = LISTWISE_PROMPTS
    else:
        raise ValueError(f"Invalid reranking strategy: {strategy}")

    if prompt_type not in templates:
        raise ValueError(f"Invalid prompt type '{prompt_type}' for strategy '{strategy}'")

    return templates[prompt_type]

def format_prompt_parts(template_parts: List[str], **kwargs) -> List[str]:
    """
    Formats the text parts of a template with provided arguments.
    Handles list expansion for listwise prompts.
    """
    formatted_parts = []
    image_placeholder = kwargs.get("image_placeholder", "<image>") # Standard placeholder

    for part in template_parts:
        # Handle listwise candidate expansion
        if "Candidate {i+1}" in part and "num_candidates" in kwargs:
            num_candidates = kwargs["num_candidates"]
            for i in range(num_candidates):
                 # Add image placeholder for each candidate
                 formatted_parts.append(f"Candidate {i+1}: {image_placeholder}. ")
        else:
            try:
                # Basic formatting for other placeholders
                formatted_part = part.format(**kwargs)
                formatted_parts.append(formatted_part)
            except KeyError as e:
                 # If a key is missing, keep the placeholder (might be intentional)
                 # print(f"Warning: Placeholder {e} not found in kwargs for part: '{part}'")
                 formatted_parts.append(part) # Add original part if formatting fails

    return formatted_parts