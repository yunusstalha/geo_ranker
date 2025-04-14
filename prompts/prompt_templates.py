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


def build_pointwise_qwen(mode = 'basic'):
        if mode == 'basic':
            conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Given this panorama street view query:"},
                    {"type": "image"},  
                    {"type": "text", "text": "And the satellite image:"},
                    {"type": "image"},
                    {"type": "text", "text": "Predict a matching score between 0 (worst) and 100 (best). Is this panorama street view taken from the same location as the satellite image?"},   
                    {"type": "text", "text": "Return your result in JSON format. Example: {\"score\": 85}"},
                    {"type": "text", "text": "Return only the JSON result, without any additional text."}
                    ]
                }
            ]
            return conversation

def build_pairwise_qwen():
    pass


def build_listwise_qwen():
    pass