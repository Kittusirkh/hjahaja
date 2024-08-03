import os
import safetensors.torch
import torch

# Function to load and extract tensors from a safetensors file
def extract_tensors(file_path, base_dir):
    data = safetensors.torch.load_file(file_path)
    extracted_files = {}
    
    for key, value in data.items():
        # Replace slashes in the key with underscores to create valid filenames
        file_name = key.replace('/', '_')
        
        # Save the tensor to the base directory with the modified filename
        tensor_path = os.path.join(base_dir, f'{file_name}.pt')
        torch.save(value, tensor_path)
        
        extracted_files[key] = tensor_path
        print(f"Saved tensor '{key}' to '{tensor_path}'")
    
    return extracted_files

# Define the base directory for extracted files
base_dir = 'save2'

# Create the base directory if it doesn't exist
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Extract tensors from sd_xl_turbo_1.0.safetensors
sd_xl_turbo_path = 'sd_xl_turbo_1.0.safetensors'
sd_xl_turbo_data = safetensors.torch.load_file(sd_xl_turbo_path)

# Extract tensors from flux1-schnell.sft
flux1_schnell_path = 'flux1-schnell.sft'
flux1_schnell_data = safetensors.torch.load_file(flux1_schnell_path)

# Save flux1-schnell.sft tensors with sd_xl_turbo_1.0.safetensors keys
for key, value in sd_xl_turbo_data.items():
    if key in flux1_schnell_data:
        new_value = flux1_schnell_data[key]
    else:
        new_value = value
        
    # Replace slashes in the key with underscores to create valid filenames
    file_name = key.replace('/', '_')
    
    # Save the tensor to the base directory with the modified filename
    tensor_path = os.path.join(base_dir, f'{file_name}.pt')
    torch.save(new_value, tensor_path)
    
    print(f"Saved tensor '{key}' to '{tensor_path}'")

print("Extraction and renaming complete. Files are organized in the base directory.")
