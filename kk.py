import torch
from safetensors import safe_save

def load_sft_model(filepath):
    # Assuming `flux1-schnell.sft` is a PyTorch model checkpoint file
    # Replace with the actual loading code based on the `.sft` format
    return torch.load(filepath)

# Path to the .sft file
sft_file_path = "flux1-schnell.sft"

# Load the model
model = load_sft_model(sft_file_path)

# Extract the state dictionary from the model
model_state_dict = model.state_dict()

# Define the path for the .safetensors file
safetensors_path = "flux1-schnell.safetensors"

# Save the state dictionary as a .safetensors file
safe_save(model_state_dict, safetensors_path)

print(f"Model saved as {safetensors_path}")
