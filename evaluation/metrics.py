# src/evaluation/metrics.py
import logging
from typing import List, Any, Dict

logger = logging.getLogger(__name__)

def calculate_accuracy_at_k(
    reranked_indices: List[Any],
    ground_truth_index: Any,
    k_list: List[int]
) -> Dict[str, float]:
    """
    Calculates Top-K accuracy (whether the ground truth is within the top K results).

    Args:
        reranked_indices (List[Any]): List of original indices sorted by the reranker.
        ground_truth_index (Any): The actual correct index for the query.
        k_list (List[int]): A list of K values to compute accuracy for (e.g., [1, 5, 10]).

    Returns:
        Dict[str, float]: Dictionary mapping "Top-K" or "Acc@K" to accuracy (1.0 or 0.0).
    """
    accuracies = {}
    found = False
    if not reranked_indices:
        logger.warning("Received empty reranked list for accuracy calculation.")
        for k in k_list:
             accuracies[f"Acc@{k}"] = 0.0
        return accuracies

    max_k = max(k_list)
    top_k_results = reranked_indices[:max_k]

    try:
        rank = top_k_results.index(ground_truth_index) + 1 # 1-based rank
        found = True
    except ValueError:
        # Ground truth not found in the top max_k results
        rank = float('inf')
        found = False

    logger.debug(f"Ground Truth Rank: {rank if found else 'Not Found in Top ' + str(max_k)}")

    for k in k_list:
        accuracies[f"Acc@{k}"] = 1.0 if rank <= k else 0.0

    return accuracies


def calculate_recall_at_k(
    reranked_indices: List[Any],
    ground_truth_indices: List[Any], # Allow multiple relevant items
    k_list: List[int]
) -> Dict[str, float]:
    """
    Calculates Recall@K (proportion of relevant items found in the top K results).

    Args:
        reranked_indices (List[Any]): List of original indices sorted by the reranker.
        ground_truth_indices (List[Any]): List of all correct/relevant indices for the query.
        k_list (List[int]): A list of K values to compute recall for (e.g., [1, 5, 10]).

    Returns:
        Dict[str, float]: Dictionary mapping "Recall@K" to recall value (0.0 to 1.0).
    """
    recalls = {}
    if not reranked_indices:
        logger.warning("Received empty reranked list for recall calculation.")
        for k in k_list:
             recalls[f"Recall@{k}"] = 0.0
        return recalls
    if not ground_truth_indices:
        logger.warning("Received empty ground truth list for recall calculation.")
        for k in k_list:
             recalls[f"Recall@{k}"] = 0.0 # Or arguably 1.0 if no relevant items exist? Let's use 0.0
        return recalls

    ground_truth_set = set(ground_truth_indices)
    num_relevant = len(ground_truth_set)

    for k in k_list:
        top_k_set = set(reranked_indices[:k])
        relevant_found_in_top_k = len(ground_truth_set.intersection(top_k_set))
        recall = relevant_found_in_top_k / num_relevant if num_relevant > 0 else 0.0
        recalls[f"Recall@{k}"] = recall
        logger.debug(f"Recall@{k}: Found {relevant_found_in_top_k}/{num_relevant}")

    return recalls


def parse_metric_config(metric_names: List[str]) -> Tuple[List[int], List[int]]:
    """Parses metric names like "Top-1", "Recall@5" into k values."""
    acc_k_list = []
    rec_k_list = []
    for name in metric_names:
        name_lower = name.lower()
        match = re.match(r"(top-|acc@)(\d+)", name_lower)
        if match:
            acc_k_list.append(int(match.group(2)))
            continue
        match = re.match(r"recall@(\d+)", name_lower)
        if match:
            rec_k_list.append(int(match.group(2)))
            continue
        logger.warning(f"Could not parse metric name: {name}. Skipping.")

    # Ensure uniqueness and sort
    acc_k_list = sorted(list(set(acc_k_list)))
    rec_k_list = sorted(list(set(rec_k_list)))

    # Compatibility: Treat Top-1 as Acc@1
    if "top-1" in [m.lower() for m in metric_names] and 1 not in acc_k_list:
         acc_k_list.append(1)
         acc_k_list.sort()


    return acc_k_list, rec_k_list