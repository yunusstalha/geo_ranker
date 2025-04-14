# reranking/prompts.py
import logging
import json # Import json for parsing examples if needed later
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# --- Common Instructions ---
JSON_INSTRUCTION = "Respond ONLY with a valid JSON object containing the required information and no other text."

# --- Define Expected JSON Output Schemas (as comments or descriptions) ---
# Pointwise Score Schema: {"score": float}  (Score between 0.0 and 10.0)
# Pointwise Reasoning+Score Schema: {"reasoning": "...", "score": float}
# Pairwise Choice Schema: {"choice": "A" | "B"}
# Listwise Rank Schema: {"ranking": List[int]} (List of 1-based indices)
# Listwise Score Schema: {"scores": List[float]} (List of scores corresponding to input order)

# --- Pointwise Prompts (Updated for JSON) ---
POINTWISE_PROMPTS = {
    # 'basic' with likelihood is tricky for JSON, maybe skip or rethink.
    # Let's focus on direct prediction prompts first.
    "score": [
        {'type': 'text', 'content': "Given this street view query:"},
        {'type': 'image_placeholder'}, # Placeholder for query image
        {'type': 'text', 'content': "Consider the satellite image:"},
        {'type': 'image_placeholder'}, # Placeholder for candidate image
        {'type': 'text', 'content': "On a scale of 0.0 to 10.0, how well does this satellite image match the query image based on visual features (buildings, roads, layout, vegetation)?"},
        {'type': 'text', 'content': "Respond with a JSON object containing a single key 'score' with the numerical score. Example: `{{\"score\": 8.5}}`. " + JSON_INSTRUCTION}
    ],
     "score_reasoning": [
        {'type': 'text', 'content': "Given this street view query:"},
        {'type': 'image_placeholder'},
        {'type': 'text', 'content': "Consider the satellite image:"},
        {'type': 'image_placeholder'},
        {'type': 'text', 'content': "Explain step-by-step why this satellite image does or does not match the query image, considering features like buildings, roads, and layout."},
        {'type': 'text', 'content': "Based on your reasoning, assign a matching score from 0.0 (no match) to 10.0 (perfect match)."},
        {'type': 'text', 'content': "Respond with a JSON object containing keys 'reasoning' (string) and 'score' (float). Example: `{{\"reasoning\": \"The road layout matches well...\", \"score\": 9.0}}`. " + JSON_INSTRUCTION}
    ]
    # "reasoning" only could output: {"reasoning": "...", "is_match": boolean | "match_level": "high"|"medium"|"low" }
}

# --- Pairwise Prompts (Updated for JSON) ---
PAIRWISE_PROMPTS = {
    "basic": [
        {'type': 'text', 'content': "Given this street view query:"},
        {'type': 'image_placeholder'}, # Query
        {'type': 'text', 'content': "Now consider two candidate satellite images. Candidate A:"},
        {'type': 'image_placeholder'}, # Candidate A
        {'type': 'text', 'content': "Candidate B:"},
        {'type': 'image_placeholder'}, # Candidate B
        {'type': 'text', 'content': "Which satellite image (A or B) is a better match for the query image?"},
        {'type': 'text', 'content': "Respond with a JSON object containing a single key 'choice' with the value 'A' or 'B'. Example: `{{\"choice\": \"A\"}}`. " + JSON_INSTRUCTION}
    ],
    "reasoning": [
        {'type': 'text', 'content': "Given this street view query:"},
        {'type': 'image_placeholder'}, # Query
        {'type': 'text', 'content': "Now consider two candidate satellite images. Candidate A:"},
        {'type': 'image_placeholder'}, # Candidate A
        {'type': 'text', 'content': "Candidate B:"},
        {'type': 'image_placeholder'}, # Candidate B
        {'type': 'text', 'content': "Compare Candidate A and Candidate B to the query image. Explain the key visual similarities and differences for each candidate relative to the query."},
        {'type': 'text', 'content': "Based on your comparison, which satellite image (A or B) is the better match?"},
        {'type': 'text', 'content': "Respond with a JSON object containing keys 'reasoning' (string comparing A and B) and 'choice' ('A' or 'B'). Example: `{{\"reasoning\": \"Candidate A has a similar building shape...\", \"choice\": \"A\"}}`. " + JSON_INSTRUCTION}
    ],
}

# --- Listwise Prompts (Updated for JSON) ---
# We still need a way to represent the list of candidates.
# Let's modify format_prompt_parts to handle this.
LISTWISE_PROMPTS = {
    "basic": [ # Basic ranking
        {'type': 'text', 'content': "Given this street view query:"},
        {'type': 'image_placeholder'}, # Query
        {'type': 'text', 'content': "Here are {num_candidates} candidate satellite images:"},
        # Marker for where the candidate list items will be inserted by format_prompt_parts
        {'type': 'candidate_list_placeholder', 'num_candidates_var': 'num_candidates'},
        {'type': 'text', 'content': "Rank these {num_candidates} satellite images from best match (most similar, rank 1) to worst match (least similar) for the query image."},
        {'type': 'text', 'content': "Respond with a JSON object containing a single key 'ranking', whose value is a list of the candidate indices (integers from 1 to {num_candidates}) in ranked order. Example for 3 candidates: `{{\"ranking\": [3, 1, 2]}}`. " + JSON_INSTRUCTION}
    ],
    "score": [ # Listwise scoring
        {'type': 'text', 'content': "Given this street view query:"},
        {'type': 'image_placeholder'}, # Query
        {'type': 'text', 'content': "Here are {num_candidates} candidate satellite images:"},
        {'type': 'candidate_list_placeholder', 'num_candidates_var': 'num_candidates'},
        {'type': 'text', 'content': "Assign a matching score from 0.0 (worst) to 10.0 (best) for each candidate satellite image (1 to {num_candidates}) compared to the query image."},
        {'type': 'text', 'content': "Respond with a JSON object containing a single key 'scores', whose value is a list of numerical scores corresponding to the candidates in their original order (Candidate 1 score, Candidate 2 score, ...). Example for 3 candidates: `{{\"scores\": [8.5, 2.1, 9.0]}}`. " + JSON_INSTRUCTION}
    ],
     "reasoning": [ # Listwise ranking with reasoning (might be too complex/long for context)
         {'type': 'text', 'content': "Given this street view query:"},
         {'type': 'image_placeholder'}, # Query
         {'type': 'text', 'content': "Here are {num_candidates} candidate satellite images:"},
         {'type': 'candidate_list_placeholder', 'num_candidates_var': 'num_candidates'},
         {'type': 'text', 'content': "For each candidate satellite image (1 to {num_candidates}), briefly explain its visual similarity or dissimilarity to the query image."},
         {'type': 'text', 'content': "Then, rank all {num_candidates} candidates from best match (most similar, rank 1) to worst match."},
         {'type': 'text', 'content': "Respond with a JSON object containing keys 'reasonings' (a list of strings, one per candidate) and 'ranking' (a list of integer indices from 1 to {num_candidates}) in ranked order. Example for 2 candidates: `{{\"reasonings\": [\"Reasoning for Cand 1...\", \"Reasoning for Cand 2...\"], \"ranking\": [2, 1]}}`. " + JSON_INSTRUCTION}
     ]
}


def get_prompt_template(strategy: str, prompt_type: str) -> List[Dict[str, Any]]:
    """
    Retrieves the appropriate prompt template structure (list of dicts).
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

    # Return a deep copy? Maybe not necessary if not modifying in place.
    return templates[prompt_type]


def format_prompt_structure(template_structure: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """
    Formats the prompt structure, replacing placeholders and expanding lists.

    Args:
        template_structure: The list of dicts defining the prompt template.
        **kwargs: Values to fill into placeholders (e.g., num_candidates).

    Returns:
        A list of dictionaries representing the final prompt structure, ready for VLM handlers.
        Example element: {'type': 'text', 'content': '...'}, {'type': 'image', 'index': 0}
    """
    final_structure = []
    image_index_counter = 0 # Track which image from the input list we are referring to

    for part in template_structure:
        part_type = part.get('type')

        if part_type == 'text':
            original_content = part.get('content', '')
            try:
                # Format the text content using provided kwargs
                formatted_content = original_content.format(**kwargs)
                final_structure.append({'type': 'text', 'content': formatted_content})
            except KeyError as e:
                logger.warning(f"Placeholder {e} not found in kwargs for text part: '{original_content}'. Keeping original.")
                final_structure.append({'type': 'text', 'content': original_content}) # Keep original on error

        elif part_type == 'image_placeholder':
            # Assign the next available image index
            final_structure.append({'type': 'image', 'index': image_index_counter})
            image_index_counter += 1

        elif part_type == 'candidate_list_placeholder':
            # Expand the listwise candidate list here
            num_candidates_var = part.get('num_candidates_var', 'num_candidates') # Key in kwargs holding the number
            if num_candidates_var in kwargs:
                num_candidates = kwargs[num_candidates_var]
                if isinstance(num_candidates, int) and num_candidates > 0:
                    for i in range(num_candidates):
                        # Add text description and image placeholder for each candidate
                        final_structure.append({'type': 'text', 'content': f"Candidate {i+1}:"})
                        final_structure.append({'type': 'image', 'index': image_index_counter})
                        image_index_counter += 1
                else:
                     logger.warning(f"Invalid 'num_candidates' value found in kwargs: {kwargs.get(num_candidates_var)}")
            else:
                logger.warning(f"Listwise placeholder found but '{num_candidates_var}' key missing in kwargs.")

        else:
            # Handle unknown part types or pass them through?
            logger.warning(f"Unknown or unhandled part type '{part_type}' in template structure.")
            # Optionally copy the part as is: final_structure.append(part.copy())

    return final_structure