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


def build_pairwise_qwen():
    pass


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
            {"type": "text", "text": "Respond ONLY with a JSON object containing the reasoning and score, like this: {\"reasoning\": \"<your_reasoning>\", \"score\": <score_value>}"}
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