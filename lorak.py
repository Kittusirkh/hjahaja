import torch
from safetensors import safe_open

# Paths to the two LoRA SafeTensors files
lora1_path = "anime-detailer-xl.safetensors"
lora2_path = "lora.safetensors"

# Dictionary to hold merged tensors
merged_lora = {}

# Load the first LoRA model
with safe_open(lora1_path, framework="torch") as f:
    for name in f.keys():
        merged_lora[name] = f.get_tensor(name)

# Load the second LoRA model and merge it with the first
with safe_open(lora2_path, framework="torch") as f:
    for name in f.keys():
        if name in merged_lora:
            # Merge by summing the weights
            merged_lora[name] += f.get_tensor(name)
        else:
            # If the tensor is not in the first LoRA, just add it
            merged_lora[name] = f.get_tensor(name)

# Save the merged LoRA model
merged_lora_path = "merged_lora.pth"
torch.save(merged_lora, merged_lora_path)

print(f"Merged LoRA model saved to {merged_lora_path}")
