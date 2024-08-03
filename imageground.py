import os
import safetensors.torch
import torch

# Load the models
model1_path = 'flux1-schnell.sft'
model2_path = 'sd_xl_turbo_1.0.safetensors'

# Load the state dictionaries
state_dict1 = safetensors.torch.load_file(model1_path)
state_dict2 = safetensors.torch.load_file(model2_path)

# Combine the state dictionaries
merged_state_dict = {**state_dict1, **state_dict2}

# Rename the keys
renamed_state_dict = {}
for key, value in merged_state_dict.items():
    new_key = 'prefix_' + key  # You can customize this renaming rule
    renamed_state_dict[new_key] = value

# Save the merged and renamed model
output_path = 'merged_model.safetensors'
safetensors.torch.save_file(renamed_state_dict, output_path)

print(f"Merged model saved to {output_path}")
