import torch
from safetensors.torch import load_file, save_file

def prune_model(lora_model, prune_rate=0.2):
    pruned_model = {}
    for key, tensor in lora_model.items():
        # Ensure tensor is in floating-point format
        if not tensor.is_floating_point():
            tensor = tensor.float()

        # Set a threshold below which weights will be zeroed out
        threshold = torch.quantile(tensor.abs(), prune_rate)
        pruned_model[key] = torch.where(tensor.abs() < threshold, torch.zeros_like(tensor), tensor)
    
    return pruned_model

# Load the LoRA model
lora_model = load_file("merged_lora.safetensors")

# Prune the model
prune_rate = 0.2  # Adjust prune rate as needed
pruned_lora_model = prune_model(lora_model, prune_rate)

# Save the pruned model
save_file(pruned_lora_model, "pruned_lora_model.safetensors")
