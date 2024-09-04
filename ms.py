import torch
from safetensors.torch import load_file, save_file

def merge_lora_models(base_path, lora_paths, output_path):
    try:
        # Load the new base model
        base_lora = load_file(base_path)
        print("Base LoRA keys:", base_lora.keys())

        # Load all other LoRA models
        loras = [load_file(path) for path in lora_paths]

        # Print the loaded keys for each additional model
        for i, lora in enumerate(loras):
            print(f"LoRA model {i+1} keys:", lora.keys())

        # Start with the base model
        merged_lora = base_lora.copy()

        # Add the weights from the other models
        for lora in loras:
            for key, tensor in lora.items():
                if key in merged_lora:
                    # Check if the shapes are the same
                    if merged_lora[key].shape == tensor.shape:
                        merged_lora[key] += tensor
                    else:
                        print(f"Skipping key '{key}' due to shape mismatch: {merged_lora[key].shape} vs {tensor.shape}")
                else:
                    # If key is not found in the base model, add it
                    merged_lora[key] = tensor

        # Optionally average the weights by the number of models
        num_models = len(loras) + 1  # Including the base model
        for key in merged_lora:
            merged_lora[key] /= num_models

        # Save the merged model
        save_file(merged_lora, output_path)
        print(f"Merged model saved to {output_path}")

    except Exception as e:
        print("Error occurred:", e)

# File paths
base_lora_path = "AWPortrait-FL-lora.safetensors"
lora_paths = [
    "FLUX-anime1.safetensors",
    "mjv6_lora.safetensors",
    "pytorch_lora_weights.safetensors",
    "anime_lora.safetensors",
    "art_lora.safetensors"
]
output_lora_path = "lora.safetensors"

# Perform the merge
merge_lora_models(base_lora_path, lora_paths, output_lora_path)
