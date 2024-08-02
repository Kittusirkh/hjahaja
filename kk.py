import torch
from safetensors import safe_load, save  # Import correct methods

def load_sft_model(filepath):
    # Implement the function to load .sft files.
    # Adjust based on how your .sft files are stored.
    return torch.load(filepath)

# Path to your .sft file
sft_file_path = "flux1-schnell.sft"

# Load the model
model = load_sft_model(sft_file_path)

# Extract the state dict from the model
model_state_dict = model.state_dict()

# Define the path for the .safetensors file
safetensors_path = "flux1-schnell.safetensors"

# Save the state dict as a .safetensors file
save(model_state_dict, safetensors_path)  # Use the correct function

print(f"Model saved as {safetensors_path}")
