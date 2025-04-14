# src/reranking/strategies.py
import logging
from typing import List, Tuple, Dict, Any
from PIL import Image
import re
import numpy as np
import itertools
import heapq # For efficient pairwise aggregation

from ..models.base_vlm import BaseVLMHandler
from .prompts import get_prompt_template, format_prompt_parts

logger = logging.getLogger(__name__)

def parse_numerical_score(text: str) -> float | None:
    """Extracts the first numerical score found in the text."""
    # Matches integers or decimals
    match = re.search(r"(\d+(\.\d+)?)", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def parse_reasoning_score(text: str, keyword: str = "Score:") -> float | None:
    """Extracts a numerical score following a specific keyword (e.g., 'Score:')."""
    lines = text.split('\n')
    for line in reversed(lines): # Check from the end
        line = line.strip()
        if line.startswith(keyword):
            score_part = line[len(keyword):].strip()
            return parse_numerical_score(score_part)
    # Fallback: check anywhere in the text if not found on a specific line
    match = re.search(rf"{keyword}\s*(\d+(\.\d+)?)", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    # Fallback: if no keyword found, try extracting any number from the last line
    if lines:
        return parse_numerical_score(lines[-1])
    return None


def parse_pairwise_choice(text: str, choice_a: str = 'A', choice_b: str = 'B', keyword: str = "Choice:") -> str | None:
    """Extracts the choice (A or B) from the text, potentially after a keyword."""
    lines = text.split('\n')
    choice = None
    # Check for keyword first
    for line in reversed(lines):
        line = line.strip()
        if line.startswith(keyword):
            choice_part = line[len(keyword):].strip()
            if choice_part.upper() == choice_a:
                return choice_a
            elif choice_part.upper() == choice_b:
                return choice_b
            else: # Found keyword but invalid choice
                 logger.warning(f"Found keyword '{keyword}' but invalid choice: '{choice_part}'")
                 return None # Explicitly return None if keyword found but choice is wrong

    # If no keyword found, check the last line directly
    if lines:
        last_line = lines[-1].strip().upper()
        if last_line == choice_a:
            return choice_a
        if last_line == choice_b:
            return choice_b

    # Fallback: Check the entire text (less reliable)
    text_upper = text.upper()
    # Be careful not to match 'A' or 'B' within words if checking whole text
    if choice_a in text_upper and choice_b not in text_upper:
        return choice_a
    if choice_b in text_upper and choice_a not in text_upper:
        return choice_b

    logger.warning(f"Could not parse pairwise choice '{choice_a}' or '{choice_b}' from text: '{text}'")
    return None # Could not determine choice

def parse_listwise_ranking(text: str, num_candidates: int, keyword: str = "Ranking:") -> List[int] | None:
    """Extracts a comma-separated list of ranked indices (1-based) from text."""
    lines = text.split('\n')
    rank_str = None

    # Look for keyword first
    for line in reversed(lines):
        line = line.strip()
        if line.startswith(keyword):
            rank_str = line[len(keyword):].strip()
            break

    # If no keyword, try the last line
    if rank_str is None and lines:
        rank_str = lines[-1].strip()

    if rank_str:
        try:
            # Remove any surrounding brackets/quotes
            rank_str = rank_str.strip("[]()'\" ")
            # Split by comma, handle potential extra spaces
            indices = [int(x.strip()) for x in rank_str.split(',')]

            # Validate the ranking
            if len(indices) != num_candidates:
                logger.warning(f"Parsed list length {len(indices)} != expected {num_candidates}. Ranking: {indices}")
                return None
            if sorted(indices) != list(range(1, num_candidates + 1)):
                logger.warning(f"Parsed list indices are invalid (not 1 to {num_candidates}). Ranking: {indices}")
                return None

            # Convert to 0-based index for internal use
            return [i - 1 for i in indices]
        except ValueError:
            logger.warning(f"Could not parse comma-separated integers from: '{rank_str}'")
            return None
        except Exception as e:
            logger.error(f"Error parsing listwise ranking '{rank_str}': {e}")
            return None

    logger.warning(f"Could not find or parse listwise ranking from text: '{text}'")
    return None

def parse_listwise_scores(text: str, num_candidates: int) -> List[float] | None:
     """Extracts a comma-separated list of scores from text."""
     lines = text.split('\n')
     score_str = lines[-1].strip() # Assume scores are on the last line for simplicity

     if score_str:
        try:
            # Remove any surrounding brackets/quotes
            score_str = score_str.strip("[]()'\" ")
            scores = [float(x.strip()) for x in score_str.split(',')]
            if len(scores) != num_candidates:
                logger.warning(f"Parsed score list length {len(scores)} != expected {num_candidates}. Scores: {scores}")
                return None
            return scores
        except ValueError:
            logger.warning(f"Could not parse comma-separated floats from: '{score_str}'")
            return None
        except Exception as e:
            logger.error(f"Error parsing listwise scores '{score_str}': {e}")
            return None

     logger.warning(f"Could not parse listwise scores from text: '{text}'")
     return None

# --- Reranking Functions ---

def rerank_pointwise(
    vlm: BaseVLMHandler,
    query_image: Image.Image,
    candidate_images: List[Image.Image],
    candidate_original_indices: List[Any], # Store original IDs/indices
    config: Dict,
    generation_args: Dict | None = None
) -> List[Tuple[Any, float]]:
    """
    Reranks candidates using pointwise scoring with a VLM.

    Returns:
        List of tuples (original_index, score), sorted by score (descending).
    """
    rerank_config = config.get("reranking", {})
    prompt_type = rerank_config.get("prompt_type", "basic")
    pointwise_config = rerank_config.get("pointwise", {})
    scoring_method = pointwise_config.get("scoring_method", "likelihood") # Or 'direct_score_prediction'

    template_parts = get_prompt_template("pointwise", prompt_type)
    scores = {} # Dict to store {original_index: score}

    logger.info(f"Starting pointwise reranking ({prompt_type}, {scoring_method}) for {len(candidate_images)} candidates.")

    for i, cand_image in enumerate(candidate_images):
        original_index = candidate_original_indices[i]
        logger.debug(f"Processing candidate {i+1} (Original Index: {original_index})")

        # Format the prompt for this specific candidate
        prompt_kwargs = {
            "query_desc": "Given this street view query",
            "image_placeholder": "<image>" # Standard placeholder
        }
        formatted_parts = format_prompt_parts(template_parts, **prompt_kwargs)

        # Prepare images: Query first, then the current candidate
        images_for_vlm = [query_image, cand_image]

        # Generate the actual prompt string using the VLM's formatter
        prompt = vlm.format_prompt(formatted_parts, images_for_vlm)

        score = 0.0 # Default score
        try:
             if scoring_method == "likelihood":
                 # Define target texts for likelihood calculation
                 # This needs careful design based on the prompt.
                 # Example for "basic" prompt:
                 target_texts = ["Relevant", "Not Relevant"] # Or "Yes", "No"; "Good Match", "Bad Match"
                 likelihoods = vlm.get_likelihood(prompt, images_for_vlm, target_texts)
                 # Assuming lower neg-log-likelihood (higher likelihood) is better
                 # Score could be likelihood of "Relevant" or difference/ratio
                 score = likelihoods[0] # Example: Use likelihood of the positive class
                 logger.debug(f"Candidate {i+1}: Likelihoods={likelihoods}, Score={score}")

             elif scoring_method == "direct_score_prediction":
                 response = vlm.generate_response(prompt, images_for_vlm, generation_args)
                 if prompt_type == "score":
                     parsed_score = parse_numerical_score(response)
                 elif prompt_type == "score_reasoning":
                      parsed_score = parse_reasoning_score(response, keyword="Score:")
                 else: # Includes "reasoning" prompt - try to extract score anyway as fallback
                      parsed_score = parse_numerical_score(response) # Less reliable

                 if parsed_score is not None:
                     score = parsed_score
                     logger.debug(f"Candidate {i+1}: Response='{response[:100]}...', Score={score}")
                 else:
                     logger.warning(f"Candidate {i+1}: Failed to parse score from response: '{response[:100]}...'")
                     score = -1.0 # Assign low score if parsing fails

             else:
                 logger.error(f"Unsupported scoring method: {scoring_method}")
                 score = -1.0

        except Exception as e:
             logger.error(f"Error processing candidate {i+1} (Original Index: {original_index}): {e}")
             score = -1.0 # Assign low score on error

        scores[original_index] = score

    # Sort by score (descending, higher score is better)
    sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    logger.info(f"Pointwise reranking finished. Top result: {sorted_results[0] if sorted_results else 'N/A'}")
    return sorted_results


def rerank_pairwise(
    vlm: BaseVLMHandler,
    query_image: Image.Image,
    candidate_images: List[Image.Image],
    candidate_original_indices: List[Any],
    config: Dict,
    generation_args: Dict | None = None
) -> List[Tuple[Any, float]]:
    """
    Reranks candidates using pairwise comparisons with a VLM.

    Returns:
        List of tuples (original_index, score), sorted by score (descending).
        Score represents win count or other aggregation metric.
    """
    rerank_config = config.get("reranking", {})
    prompt_type = rerank_config.get("prompt_type", "basic")
    pairwise_config = rerank_config.get("pairwise", {})
    aggregation = pairwise_config.get("aggregation", "counting")

    template_parts = get_prompt_template("pairwise", prompt_type)
    num_candidates = len(candidate_images)
    if num_candidates < 2:
        logger.warning("Pairwise reranking requires at least 2 candidates.")
        # Return original order with dummy scores if only 1 candidate
        return [(idx, 0.0) for idx in candidate_original_indices]

    logger.info(f"Starting pairwise reranking ({prompt_type}, {aggregation}) for {num_candidates} candidates.")

    # --- Aggregation Setup ---
    # Simple win counting
    wins = {original_index: 0 for original_index in candidate_original_indices}

    # TODO: Implement more sophisticated aggregation like Tournament Sort or other methods if needed.

    # --- Perform Comparisons ---
    # Iterate through all unique pairs (indices i, j where i < j)
    for i, j in itertools.combinations(range(num_candidates), 2):
        cand_a_image = candidate_images[i]
        cand_b_image = candidate_images[j]
        original_index_a = candidate_original_indices[i]
        original_index_b = candidate_original_indices[j]

        logger.debug(f"Comparing Candidate {i+1} (Idx: {original_index_a}) vs Candidate {j+1} (Idx: {original_index_b})")

        # Format prompt for this pair
        prompt_kwargs = {
            "query_desc": "Given this street view query",
            "image_placeholder": "<image>"
        }
        formatted_parts = format_prompt_parts(template_parts, **prompt_kwargs)
        images_for_vlm = [query_image, cand_a_image, cand_b_image]
        prompt = vlm.format_prompt(formatted_parts, images_for_vlm)

        try:
            response = vlm.generate_response(prompt, images_for_vlm, generation_args)
            choice = parse_pairwise_choice(response, choice_a='A', choice_b='B', keyword="Choice:" if prompt_type=="reasoning" else None)

            if choice == 'A':
                logger.debug(f"  -> Result: A ({original_index_a}) wins")
                wins[original_index_a] += 1
            elif choice == 'B':
                logger.debug(f"  -> Result: B ({original_index_b}) wins")
                wins[original_index_b] += 1
            else:
                logger.warning(f"  -> Result: Could not determine winner from response: '{response[:100]}...'")
                # Optional: Penalize both or neither? Currently, no score change.

        except Exception as e:
            logger.error(f"Error processing pair ({original_index_a}, {original_index_b}): {e}")

    # --- Final Sorting ---
    # Sort by win count (descending)
    sorted_results = sorted(wins.items(), key=lambda item: item[1], reverse=True)

    logger.info(f"Pairwise reranking finished. Top result: {sorted_results[0] if sorted_results else 'N/A'}")
    return sorted_results


def rerank_listwise(
    vlm: BaseVLMHandler,
    query_image: Image.Image,
    candidate_images: List[Image.Image],
    candidate_original_indices: List[Any],
    config: Dict,
    generation_args: Dict | None = None
) -> List[Tuple[Any, float]]:
    """
    Reranks candidates using a listwise approach with a VLM.
    Handles potential sliding window if context is limited.

    Returns:
        List of tuples (original_index, score). Score is position-based (lower is better)
        or parsed score if prompt_type is 'score'. Sorted by rank/score.
    """
    rerank_config = config.get("reranking", {})
    prompt_type = rerank_config.get("prompt_type", "basic")
    listwise_config = rerank_config.get("listwise", {})
    window_size = listwise_config.get("window_size") # None means full list

    num_candidates = len(candidate_images)
    template_parts = get_prompt_template("listwise", prompt_type)

    logger.info(f"Starting listwise reranking ({prompt_type}) for {num_candidates} candidates.")

    if window_size and window_size < num_candidates:
        logger.warning(f"Listwise sliding window not fully implemented. Processing full list.")
        # TODO: Implement sliding window logic if needed. Requires complex prompt adaptation
        # and aggregation of rankings/scores across windows. For now, process full list.

    prompt_kwargs = {
        "query_desc": "Given this street view query",
        "image_placeholder": "<image>",
        "num_candidates": num_candidates
    }
    # format_prompt_parts handles the candidate loop internally now
    formatted_parts = format_prompt_parts(template_parts, **prompt_kwargs)

    images_for_vlm = [query_image] + candidate_images
    prompt = vlm.format_prompt(formatted_parts, images_for_vlm)

    final_ranking = [] # List of original indices in the new order
    scores = {} # Dict {original_index: score} if using score prompt

    try:
        response = vlm.generate_response(prompt, images_for_vlm, generation_args)

        if prompt_type == "score":
            parsed_scores = parse_listwise_scores(response, num_candidates)
            if parsed_scores:
                for i, score in enumerate(parsed_scores):
                    scores[candidate_original_indices[i]] = score
                # Sort by score descending
                sorted_by_score = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                final_ranking = [idx for idx, score in sorted_by_score]
            else:
                 logger.error(f"Failed to parse listwise scores from response: {response}")

        else: # Basic or Reasoning prompt types expect ranked indices
            parsed_indices = parse_listwise_ranking(response, num_candidates, keyword="Ranking:" if prompt_type=="reasoning" else None)
            if parsed_indices:
                 # parsed_indices are 0-based indices relative to the candidate_images list
                 # Map them back to the original indices
                 final_ranking = [candidate_original_indices[idx] for idx in parsed_indices]
                 logger.debug(f"Parsed ranking (0-based): {parsed_indices}")
                 logger.debug(f"Final ranking (original indices): {final_ranking}")
            else:
                 logger.error(f"Failed to parse listwise ranking from response: {response}")

    except Exception as e:
        logger.error(f"Error during listwise generation/parsing: {e}")

    # If parsing failed or ranking is incomplete, return original order as fallback
    if not final_ranking and not scores:
        logger.warning("Listwise reranking failed. Returning original order.")
        # Assign dummy scores based on original order (lower score is better rank)
        results = [(candidate_original_indices[i], float(i)) for i in range(num_candidates)]
    elif scores: # We have scores, return sorted by score
         results = sorted(scores.items(), key=lambda item: item[1], reverse=True) # Higher score is better
    else: # We have a ranking, assign scores based on rank position
         # Assign score based on rank (lower is better, e.g., rank 0 = best)
         results = [(original_index, float(rank)) for rank, original_index in enumerate(final_ranking)]


    logger.info(f"Listwise reranking finished. Top result index: {results[0][0] if results else 'N/A'}")
    # Ensure result format matches others: List[Tuple[Any, float]] sorted by score/rank
    # If using rank-based score, reverse=False for sorting (lower rank is better)
    # If using VLM scores, reverse=True
    sort_reverse = (prompt_type == "score") # Higher score is better only if we parsed scores
    results.sort(key=lambda item: item[1], reverse=sort_reverse)

    return results


def get_rerank_function(strategy: str):
    """Returns the appropriate reranking function based on the strategy string."""
    if strategy == "pointwise":
        return rerank_pointwise
    elif strategy == "pairwise":
        return rerank_pairwise
    elif strategy == "listwise":
        return rerank_listwise
    else:
        raise ValueError(f"Unknown reranking strategy: {strategy}")