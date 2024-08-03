import os
import safetensors.torch
import torch

# Load the safetensors file
file_path = 'flux1-schnell.sft'
data = safetensors.torch.load_file(file_path)

# Define the base directory for extracted files
base_dir = 'save2'

# Create the base directory if it doesn't exist
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Iterate over the items in the safetensors file
for key, value in data.items():
    # Define the folder path for this item
    folder_path = os.path.join(base_dir, key)

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the tensor to the folder
    tensor_path = os.path.join(folder_path, f'{key}.pt')
    torch.save(value, tensor_path)

    print(f"Saved tensor '{key}' to '{tensor_path}'")

print("Extraction complete. Files are organized in folders.")
