import torch
from safetensors import safe_open, safe_save
import os

# Paths to the two LoRA SafeTensors files
lora1_path = "anime-detailer-xl.safetensors"
lora2_path = "lora.safetensors"

# Load the first LoRA model
lora1 = {}
with safe_open(lora1_path, framework="torch") as f:
    for name in f.keys():
        lora1[name] = f.get_tensor(name)

# Load the second LoRA model and merge it with the first
with safe_open(lora2_path, framework="torch") as f:
    for name in f.keys():
        if name in lora1:
            # Merge by summing the weights
            lora1[name] += f.get_tensor(name)
        else:
            # If the tensor is not in the first LoRA, just add it
            lora1[name] = f.get_tensor(name)

# Save the merged LoRA model
merged_lora_path = "merged_lora.safetensors"
safe_save(lora1, merged_lora_path)

print(f"Merged LoRA model saved to {merged_lora_path}")
