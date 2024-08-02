import torch
from diffusers import DiffusionPipeline

# Load the first model
model_1 = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/1"
).to("cuda" if torch.cuda.is_available() else "cpu")

# Load the second model
model_2 = DiffusionPipeline.from_pretrained(
    "enhanceaiteam/ImageGroundV0.1",
    torch_dtype=torch.bfloat16
).to("cuda" if torch.cuda.is_available() else "cpu")

# Extract state dicts from both models
state_dict_1 = model_1.unet.state_dict()
state_dict_2 = model_2.unet.state_dict()

# Function to merge weights
def merge_weights(weights1, weights2, alpha=0.5):
    merged_weights = {}
    for key in weights1.keys():
        if key in weights2:
            # Average the weights
            merged_weights[key] = alpha * weights1[key] + (1 - alpha) * weights2[key]
        else:
            merged_weights[key] = weights1[key]
    return merged_weights

# Merge the weights
merged_weights = merge_weights(state_dict_1, state_dict_2)

# Create a new pipeline instance with one of the model's configuration
new_model = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
).to("cuda" if torch.cuda.is_available() else "cpu")

# Load the merged weights into the new model
new_model.unet.load_state_dict(merged_weights)

# Save the merged model
save_path = "path/to/save/merged_model"
new_model.save_pretrained(save_path)

print(f"Merged model saved to {save_path}")
