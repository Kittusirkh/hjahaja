import os
import safetensors.torch
import torch

# Function to load and extract tensors from a safetensors file
def extract_tensors(file_path):
    return safetensors.torch.load_file(file_path)

# Function to save tensors to a new safetensors file
def save_tensors_to_safetensors(tensors, output_path):
    safetensors.torch.save_file(tensors, output_path)

# Function to rename keys by removing 'enhanceai' and save tensors
def process_tensors_and_save(tensors1, tensors2, base_dir, output_path):
    combined_tensors = {}
    
    # Combine tensors from both files
    for key, value in tensors1.items():
        combined_tensors[key] = value
    
    for key, value in tensors2.items():
        if key not in combined_tensors:
            combined_tensors[key] = value

    # Rename keys by removing 'enhanceai'
    renamed_tensors = {}
    for key, value in combined_tensors.items():
        new_key = key.replace('enhanceai', '')
        renamed_tensors[new_key] = value
        
        # Save the tensor to the base directory with the modified filename
        tensor_path = os.path.join(base_dir, f'{new_key}.pt')
        torch.save(value, tensor_path)
        print(f"Saved tensor '{new_key}' to '{tensor_path}'")
    
    # Save all tensors to the output safetensors file
    save_tensors_to_safetensors(renamed_tensors, output_path)
    print(f"Saved all tensors to '{output_path}'")

# Define the base directory for extracted files and the output file path
base_dir = 'save3'
output_path = 'output.safetensors'

# Create the base directory if it doesn't exist
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Extract tensors from files
sd_xl_turbo_path = 'sd_xl_turbo_1.0.safetensors'
flux1_schnell_path = 'flux1-schnell.sft'

sd_xl_turbo_data = extract_tensors(sd_xl_turbo_path)
flux1_schnell_data = extract_tensors(flux1_schnell_path)

# Process tensors, rename keys, and save to output file
process_tensors_and_save(sd_xl_turbo_data, flux1_schnell_data, base_dir, output_path)

print("Extraction, renaming, and saving complete. Output file created.")
