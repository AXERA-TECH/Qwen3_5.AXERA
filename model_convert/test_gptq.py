import inspect

from gptqmodel import GPTQModel
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer

model_path = "../../Qwen/Qwen3.5-0.8B/"
quant_path = "../../Qwen/Qwen3.5-0.8B-GPTQ-Int4-EN"
device = "cuda" if torch.cuda.is_available() else "cpu"

# test post-quant inference
model = GPTQModel.load(quant_path, device_map={"": device})
forward_args = set(inspect.signature(model.model.forward).parameters.keys())
supports_vision = "pixel_values" in forward_args
assert supports_vision, "quantized model does not support vision input, cannot run vision test"
print("loaded_model_class:", model.model.__class__.__name__)
print("loaded_model_type:", model.model.config.model_type)
print("supports_vision_generate:", supports_vision)


processor = AutoProcessor.from_pretrained(model_path)
img = Image.open("./demo.jpeg").resize((384, 384))
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Describe the image"},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(device)
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)


print(output_text)
