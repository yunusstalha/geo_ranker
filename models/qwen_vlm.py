# models/qwen_vlm.py

import os
from PIL import Image
import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TorchAoConfig
)
from .base_vlm import BaseVLM


class QwenVLM(BaseVLM):
    def __init__(self, 
                model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                use_bf16: bool = True,
                quantization: str = "int4_weight_only",
                group_size: int = 128,
                device_map: str = "auto",
                ):
        """
        Initialize the QwenVLM model and processor.
        """
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        quantization_config = TorchAoConfig(quantization, group_size=group_size)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            quantization_config=quantization_config,
        )
        self.model.eval()

def generate(self, conversation, image_inputs, max_new_tokens: int = 256) -> str:
    """
    Generate a response for the given conversation and image inputs.
    """
    # Process the conversation and image inputs
    self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
    inputs = self.processor(
        text=[prompt],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    )

    output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    print("\nModel output:")
    print(output_text[0])