import torch
from safetensors.torch import load_file, save_file

def merge_multiple_lora_models(lora_paths, output_path):
    try:
        # Load all LoRA models
        loras = [load_file(path) for path in lora_paths]

        # Print the loaded keys for each model
        for i, lora in enumerate(loras):
            print(f"LoRA model {i+1} keys:", lora.keys())

        # Initialize the merged model with the first LoRA model
        merged_lora = loras[0].copy()

        # Sum the weights of all models
        for key in merged_lora:
            for lora in loras[1:]:
                if key in lora:
                    merged_lora[key] += lora[key]
                else:
                    print(f"Key {key} not found in one of the models.")

        # Average the weights by dividing by the number of models
        num_models = len(loras)
        for key in merged_lora:
            merged_lora[key] /= num_models

        # Save the merged model
        save_file(merged_lora, output_path)
        print(f"Merged model saved to {output_path}")

    except Exception as e:
        print("Error occurred:", e)

# File paths
lora_paths = [
    "text_encoder_2.safetensors",
    "FLUX-anime1.safetensors",
    "mjv6_lora.safetensors",
    "pytorch_lora_weights.safetensors",
    "anime_lora.safetensors",
    "art_lora.safetensors"
]

output_lora_path = "lora.safetensors"

# Perform the merge
merge_multiple_lora_models(lora_paths, output_lora_path)
