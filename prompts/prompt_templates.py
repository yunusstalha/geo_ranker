# prompts/prompt_templates.py

"""
Minimal library of prompt “formats” for different experiment styles:
- pointwise
    - multiple approaches like basic, + reasoning
- pairwise
    - multiple approaches like basic, + reasoning, + extra score prediction
- listwise
    - multiple approaches like basic, + reasoning, + extra score prediction

We provide Qwen-styled conversation structures as examples. 
You could also add parallel LLaVA-specific prompt builders if needed.
"""


def build_pointwise_qwen(mode='basic'):
    """
    Builds a pointwise prompt for Qwen-VL.
    Assumes 2 images: image[0] is query, image[1] is candidate.
    """
    if mode == 'basic':
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the ground-level panorama query image:"},
                    {"type": "image"}, # Placeholder for query image (index 0)
                    {"type": "text", "text": "Here is a candidate satellite image:"},
                    {"type": "image"}, # Placeholder for candidate image (index 1)
                    {"type": "text", "text": "Evaluate if the satellite image corresponds to the location shown in the ground-level panorama."},
                    {"type": "text", "text": "Provide a confidence score between 0 (no match) and 100 (perfect match)."},
                    {"type": "text", "text": "Respond ONLY with a JSON object containing the score, like this: {\"score\": <score_value>}"}
                ]
            }
            # Qwen expects an empty assistant message to signal generation turn,
            # but apply_chat_template with add_generation_prompt=True handles this.
        ]
        return conversation
    elif mode == 'reasoning':
         conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the ground-level panorama query image:"},
                    {"type": "image"}, # Placeholder for query image (index 0)
                    {"type": "text", "text": "Here is a candidate satellite image:"},
                    {"type": "image"}, # Placeholder for candidate image (index 1)
                    {"type": "text", "text": "Evaluate if the satellite image corresponds to the location shown in the ground-level panorama."},
                    {"type": "text", "text": "First, provide a brief step-by-step reasoning comparing key visual features (e.g., road layout, building shapes, landmarks)."},
                    {"type": "text", "text": "Then, provide a confidence score between 0 (no match) and 100 (perfect match)."},
                    {"type": "text", "text": "Respond ONLY with a JSON object containing the reasoning and score, like this: {\"reasoning\": \"<your_reasoning>\", \"score\": <score_value>}"}
                ]
            }
        ]
         return conversation
    else:
        raise ValueError(f"Unknown pointwise mode: {mode}")


def build_pairwise_qwen(mode='basic'):
    """
    Builds a pairwise prompt for Qwen-VL.
    Assumes 3 images: image[0] is query, image[1] is candidate 1, image[2] is candidate 2.
    """
    # Define content based on mode
    task_description = ""
    response_format = ""
    if mode == 'basic':
        task_description = "Compare Satellite Image 1 and Satellite Image 2 to the ground-level query image. Which satellite image (1 or 2) shows the same location as the query image?"
        response_format = "Respond ONLY with a JSON object indicating the preferred image number (1 or 2), like this: {\"preference\": <1_or_2>}"
    elif mode == 'reasoning':
        task_description = "Compare Satellite Image 1 and Satellite Image 2 to the ground-level query image. Which satellite image (1 or 2) shows the same location as the query image? Provide a brief step-by-step reasoning for your choice based on visual features."
        response_format = "Respond ONLY with a JSON object containing the preference (1 or 2) and reasoning, like this: {\"preference\": <1_or_2>, \"reasoning\": \"<your_reasoning>\"}"
    else:
        raise ValueError(f"Unknown pairwise mode: {mode}")

    # Build conversation structure
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the ground-level panorama query image:"},
                {"type": "image"}, # Placeholder for query image (index 0)
                {"type": "text", "text": "Here is Satellite Image 1 (Candidate 1):"},
                {"type": "image"}, # Placeholder for candidate 1 (index 1)
                {"type": "text", "text": "Here is Satellite Image 2 (Candidate 2):"},
                {"type": "image"}, # Placeholder for candidate 2 (index 2)
                {"type": "text", "text": task_description},
                {"type": "text", "text": response_format}
            ]
        }
    ]
    return conversation



def build_listwise_qwen():
    pass

def build_pointwise_llava(mode='basic'):
    """
    Builds a pointwise prompt conversation structure for Llava-1.5.
    Assumes 2 images: image[0] is query, image[1] is candidate.
    The LlavaVLM class will format this into the final prompt string.
    """
    content = []
    if mode == 'basic':
        content = [
            {"type": "text", "text": "Here is the ground-level panorama query image:"},
            {"type": "image"}, # Placeholder for query image (index 0)
            {"type": "text", "text": "Here is a candidate satellite image:"},
            {"type": "image"}, # Placeholder for candidate image (index 1)
            {"type": "text", "text": "Evaluate if the satellite image corresponds to the location shown in the ground-level panorama."},
            {"type": "text", "text": "Provide a confidence score between 0 (no match) and 100 (perfect match)."},
            {"type": "text", "text": "Respond ONLY with a JSON object containing the score, like this: {\"score\": <score_value>}"}
        ]
    elif mode == 'reasoning':
        content = [
            {"type": "text", "text": "Here is the ground-level panorama query image:"},
            {"type": "image"}, # Placeholder for query image (index 0)
            {"type": "text", "text": "Here is a candidate satellite image:"},
            {"type": "image"}, # Placeholder for candidate image (index 1)
            {"type": "text", "text": "Evaluate if the satellite image corresponds to the location shown in the ground-level panorama."},
            {"type": "text", "text": "First, provide a brief step-by-step reasoning comparing key visual features (e.g., road layout, building shapes, landmarks)."},
            {"type": "text", "text": "Then, provide a confidence score between 0 (no match) and 100 (perfect match)."},
            {"type": "text", "text": "Respond ONLY with a JSON object containing the reasoning and score (only the value), like this: {\"reasoning\": \"<your_reasoning>\", \"score\": <score_value>}"}
        ]
    else:
        raise ValueError(f"Unknown pointwise mode: {mode}")

    # Structure for LLaVA: a single user turn with text and image markers
    conversation = [
        {
            "role": "user",
            "content": content
        }
        # The LlavaVLM prompt builder adds "ASSISTANT:" turn
    ]
    return conversation # Return the list structure

def build_pairwise_llava(mode='basic'):
    """
    Builds a pairwise prompt conversation structure for Llava-1.5.
    Assumes 3 images: image[0] is query, image[1] is candidate 1, image[2] is candidate 2.
    """
    # Define content based on mode
    task_description = ""
    response_format = ""
    if mode == 'basic':
        task_description = "Compare Satellite Image 1 and Satellite Image 2 to the ground-level query image. Which satellite image (1 or 2) shows the same location as the query image?"
        response_format = "Respond ONLY with a JSON object indicating the preferred image number (1 or 2), like this: {\"preference\": <1_or_2>}"
    elif mode == 'reasoning':
        task_description = "Compare Satellite Image 1 and Satellite Image 2 to the ground-level query image. Which satellite image (1 or 2) shows the same location as the query image? Provide a brief step-by-step reasoning for your choice based on visual features."
        response_format = "Respond ONLY with a JSON object containing the preference (1 or 2) and reasoning, like this: {\"preference\": <1_or_2>, \"reasoning\": \"<your_reasoning>\"}"
    else:
        raise ValueError(f"Unknown pairwise mode: {mode}")

    # Build content list for LLaVA conversation structure
    content = [
        {"type": "text", "text": "Here is the ground-level panorama query image:"},
        {"type": "image"}, # Placeholder for query image (index 0)
        {"type": "text", "text": "Here is Satellite Image 1 (Candidate 1):"},
        {"type": "image"}, # Placeholder for candidate 1 (index 1)
        {"type": "text", "text": "Here is Satellite Image 2 (Candidate 2):"},
        {"type": "image"}, # Placeholder for candidate 2 (index 2)
        {"type": "text", "text": task_description},
        {"type": "text", "text": response_format}
    ]

    # Structure for LLaVA: a single user turn
    conversation = [{"role": "user", "content": content}]
    return conversation
    
def get_prompt_builder(vlm_type: str, strategy: str):
    """ Returns the correct prompt builder function based on VLM and strategy. """
    builder_map = {
        ('qwen', 'pointwise'): build_pointwise_qwen,
        ('llava', 'pointwise'): build_pointwise_llava,
        ('qwen', 'pairwise'): build_pairwise_qwen,
        ('llava', 'pairwise'): build_pairwise_llava,
        # Add listwise mappings here when implemented
    }
    builder = builder_map.get((vlm_type, strategy))
    if builder is None:
        raise ValueError(f"No prompt builder found for VLM type '{vlm_type}' and strategy '{strategy}'")
    return builder