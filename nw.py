import torch
from safetensors.torch import load_file, save_file

def pad_tensor_to_match(tensor_a, tensor_b):
    # Get the shape of both tensors
    shape_a = torch.tensor(tensor_a.shape)
    shape_b = torch.tensor(tensor_b.shape)
    
    # Determine the larger shape
    max_shape = torch.max(shape_a, shape_b).tolist()
    
    # Pad both tensors to the same size as the max_shape
    padded_a = torch.nn.functional.pad(tensor_a, (0, max_shape[-1] - shape_a[-1]))
    padded_b = torch.nn.functional.pad(tensor_b, (0, max_shape[-1] - shape_b[-1]))
    
    return padded_a, padded_b

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
                
                # If shapes are incompatible, pad the tensors to match the larger size
                if base_tensor.shape != lora_tensor.shape:
                    base_tensor, lora_tensor = pad_tensor_to_match(base_tensor, lora_tensor)
                
                # Merge using the ratio
                merged_lora[key] = base_tensor * (1 - ratio) + lora_tensor * ratio
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
