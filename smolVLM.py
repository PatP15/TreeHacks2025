import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import PIL

DEVICE = "cuda" if torch.cuda.is_available() else "mps"

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
)
print("Loaded model and processor")
# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"}
        ]
    },
]
print(messages)
image_path = "./pic.jpeg"

print("Processing image...")
image = PIL.Image.open(image_path)
# Preprocess
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")

print("Generating caption...")
# Generate
generated_ids = model.generate(**inputs, max_new_tokens=50)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
print(generated_texts)