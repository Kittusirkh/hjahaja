import torch
from safetensors.torch import load_file, save_file

# Load the LoRA model
lora_model = load_file("merged_lora.safetensors")

# Quantize to 16-bit (half precision)
for key, tensor in lora_model.items():
    lora_model[key] = tensor.half()

# Prune the model
prune_rate = 0.2  # Adjust prune rate for desired size
for key, tensor in lora_model.items():
    threshold = torch.quantile(tensor.abs(), prune_rate)
    lora_model[key] = torch.where(tensor.abs() < threshold, torch.zeros_like(tensor), tensor)

# Save the reduced model
save_file(lora_model, "reduced_lora_model.safetensors")
