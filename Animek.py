import torch
from safetensors.torch import load_file, save_file

def merge_lora_models(lora_path1, lora_path2, output_path):
    try:
        # Load both LoRA models
        lora1 = load_file(lora_path1)
        lora2 = load_file(lora_path2)

        print(f"LoRA model 1 keys: {lora1.keys()}")
        print(f"LoRA model 2 keys: {lora2.keys()}")

        # Start with the first model's weights
        merged_lora = lora1.copy()

        # Merge with the second model's weights
        for key, tensor in lora2.items():
            if key in merged_lora:
                if merged_lora[key].shape == tensor.shape:
                    merged_lora[key] += tensor
                else:
                    print(f"Skipping key '{key}' due to shape mismatch: {merged_lora[key].shape} vs {tensor.shape}")
            else:
                # Add the key if it doesn't exist in the first model
                merged_lora[key] = tensor

        # Save the merged model
        save_file(merged_lora, output_path)
        print(f"Merged LoRA model saved to {output_path}")

    except Exception as e:
        print("Error occurred:", e)

# File paths for the two LoRA models to merge
lora_model_1_path = "fluxanimes.safetensors"
lora_model_2_path = "FLUX-anime2.safetensors"
output_lora_path = "fluxanimes.safetensors"

# Merge the two LoRA models
merge_lora_models(lora_model_1_path, lora_model_2_path, output_lora_path)
