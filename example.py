import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TorchAoConfig
import requests
from PIL import Image

# def load_image_from_url(url: str) -> Image.Image:
#     response = requests.get(url, stream=True)
#     response.raise_for_status()
#     return Image.open(response.raw).convert("RGB")
def load_image(path: str) -> Image.Image:
    """
    Load an image from a local path.
    """
    return Image.open(path).convert("RGB")
def plot_query_and_images(query_image: Image.Image, image1: Image.Image, image2: Image.Image):
    """
    Plot the query image and two images side by side.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(query_image)
    axes[0].set_title("Query Image")
    axes[0].axis("off")
    
    axes[1].imshow(image1)
    axes[1].set_title("Image 1")
    axes[1].axis("off")
    
    axes[2].imshow(image2)
    axes[2].set_title("Image 2")
    axes[2].axis("off")
    
    plt.savefig("query_and_images.png")

def main():
    # Check if CUDA is available and set device accordingly.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantization_config = TorchAoConfig("int4_weight_only", group_size=128)

    # Load the Qwen2.5-VL model and processor.
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        # device_map={"": 0},
        device_map="auto",
        torch_dtype=torch.bfloat16,  # Use bfloat16 if available, else use torch.float16.
        quantization_config=quantization_config,
    )
    
    processor = AutoProcessor.from_pretrained(model_id)

    # Define two image URLs for testing.
    image_url1 = "/research/nfs_yilmaz_15/yunus/data/VIGOR/SanFrancisco/panorama/ScGwgVhnS9mLGl47wRnicQ,37.752695,-122.437849,.jpg"
    image_url2 = "/research/nfs_yilmaz_15/yunus/data/VIGOR/SanFrancisco/satellite/satellite_37.752731622631885_-122.43810708310743.png"
    image_url3 = "/research/nfs_yilmaz_15/yunus/data/VIGOR/SanFrancisco/satellite/satellite_37.752731622631885_-122.43769208587767.png"

    print("Loading images...")
    image1 = load_image(image_url1)
    image2 = load_image(image_url2)
    image3 = load_image(image_url3)

    # Create a conversation with two images and a text prompt.
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Given this panorama street view query:"},
                {"type": "image"},  
                {"type": "text", "text": "And two satellite images. Sat Image 1:"},
                {"type": "image"},
                {"type": "text", "text": " Sat Image 2:"},
                {"type": "image"},
                {"type": "text", "text": "Which one of the satellite images is the best match for the street view? Sat 1 or Sat 2?"},
            ]
        }
    ]
    
    # Prepare the text prompt using the processor's chat template
    # Setting tokenize=False here to get a full string prompt.
    print("Preparing conversation prompt...")
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Process the images.
    # Here we assume the processor can handle the images separately.
    # If your images are in URL form, and the processor supports a "url" key,
    # you might alternatively pass them directly. Otherwise, load them as objects.
    image_inputs = [image1, image2, image3]
    
    # Create the model inputs.
    # The processor will take the text and list of images.
    print("Creating model inputs...")
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    )
    # Run inference.
    print("Running inference...")
    output_ids = model.generate(**inputs, max_new_tokens=2048)
    
    # Trim the generated output if necessary.
    # We assume that the input prompt tokens are at the beginning.
    # Here we calculate the extra tokens generated.
    generated_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    
    # Decode the model output.
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    print("\nModel output:")
    print(output_text[0])

if __name__ == "__main__":
    main()
