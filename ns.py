import torch
from safetensors.torch import load_file, save_file

def merge_lora_models(base_path, lora_path, output_path, ratio):
    try:
        # Load the base and LoRA models
        base_lora = load_file(base_path)
        lora_2 = load_file(lora_path)

        # Print the loaded keys
        print("Base LoRA keys:", base_lora.keys())
        print("2nd LoRA keys:", lora_2.keys())

        # Merge the models with the given ratio
        merged_lora = {}
        for key in base_lora:
            if key in lora_2:
                base_tensor = base_lora[key]
                lora_tensor = lora_2[key]
                
                # Print tensor shapes
                print(f"Key: {key}, Base shape: {base_tensor.shape}, LoRA shape: {lora_tensor.shape}")
                
                # Check if shapes are compatible for merging
                if base_tensor.shape == lora_tensor.shape:
                    # Merge using the ratio
                    merged_lora[key] = base_tensor * (1 - ratio) + lora_tensor * ratio
                else:
                    print(f"Skipping key '{key}' due to incompatible shapes.")
                    merged_lora[key] = base_tensor
            else:
                merged_lora[key] = base_lora[key]

        for key in lora_2:
            if key not in merged_lora:
                merged_lora[key] = lora_2[key]

        # Save the merged model
        save_file(merged_lora, output_path)
        print(f"Merged model saved to {output_path}")

    except Exception as e:
        print("Error occurred:", e)

# File paths
base_lora_path = "nsfw_flux_lora_v1.safetensors"
lora_2_path = "NSFW_master.safetensors"
output_lora_path = "nsfwlora.safetensors"

# Set the ratio for merging
merge_ratio = 0.25

# Perform the merge
merge_lora_models(base_lora_path, lora_2_path, output_lora_path, merge_ratio)
