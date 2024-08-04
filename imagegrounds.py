import torch
from safetensors import safe_open, safe_save

# Function to load a model from safetensors format
def load_model(path):
    with safe_open(path, framework="pt") as f:
        state_dict = {}
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict

# Function to save a model in safetensors format
def save_model(state_dict, path):
    safe_save(state_dict, path)

# Load the weights of both models
model1_path = 'sd_xl_turbo_1.0.safetensors'
model2_path = 'flux1-schnell.sft'

state_dict1 = load_model(model1_path)
state_dict2 = load_model(model2_path)

# Initialize a new state dictionary for the combined model
combined_state_dict = {}

# Ensure both models have the same keys
keys1 = set(state_dict1.keys())
keys2 = set(state_dict2.keys())

if keys1 != keys2:
    raise ValueError("Models have different structures and cannot be directly combined.")

# Combine the weights by averaging
for key in keys1:
    combined_state_dict[key] = (state_dict1[key] + state_dict2[key]) / 2

# Save the combined weights
combined_model_path = 'combined_model.safetensors'
save_model(combined_state_dict, combined_model_path)

print(f"Combined model saved to '{combined_model_path}'")
