from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Create a directory for SSD offloading if it doesn't exist
offload_dir = "./offload_weights"
if not os.path.exists(offload_dir):
    os.makedirs(offload_dir)

# model_path = "/home/usama/.lmstudio/models/lmstudio-community/gemma-4-E2B-it-GGUF/"
model_path = "/home/usama/.lmstudio/models/unsloth/gemma-4-26B-A4B-it-GGUF/"
gguf_file = "gemma-4-26B-A4B-it-UD-IQ2_M.gguf"
tokenizer_id = "google/gemma-4-26B-A4B-it"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

# Optimized Loading
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    gguf_file=gguf_file,
    device_map="auto",
    # Use the SSD as temporary RAM  because we have 64GB ram..
    offload_folder=offload_dir,
    # Prevents loading the whole model into RAM at once
    low_cpu_mem_usage=True,
    # Use bfloat16 to halve peak RAM during GGUF dequantization
    # (128 experts x 30 layers would be ~95GB in float32, ~47GB in bfloat16)
    torch_dtype=torch.bfloat16
)

messages = [{"role": "user", "content": "Write a short poem about a robot learning to paint."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.7,
    do_sample=True
)

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))