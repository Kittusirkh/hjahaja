import torch
from transformers import DiffusionPipeline

# Load the pre-trained models
model_1 = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
model_2 = DiffusionPipeline.from_pretrained("enhanceaiteam/ImageGround0.1")

# Extract the model weights
model_1_weights = model_1.model.state_dict()
model_2_weights = model_2.model.state_dict()

# Function to merge weights
def merge_weights(weights1, weights2, alpha=0.5):
    merged_weights = {}
    for key in weights1.keys():
        if key in weights2:
            merged_weights[key] = alpha * weights1[key] + (1 - alpha) * weights2[key]
        else:
            merged_weights[key] = weights1[key]
    return merged_weights

# Merge the weights
merged_weights = merge_weights(model_1_weights, model_2_weights)
print(merged_weights)

# Load a new model architecture or use one of the existing ones
base_model_name = "black-forest-labs/FLUX.1-schnell"  # or use another base model name
new_model = DiffusionPipeline.from_pretrained(base_model_name)

# Apply the merged weights to the new model
new_model.model.load_state_dict(merged_weights)

# Save the merged model
save_path = "path/to/save/merged_model"
new_model.save_pretrained(save_path)

print(f"Merged model saved to {save_path}")
