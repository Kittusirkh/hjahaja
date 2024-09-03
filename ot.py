import torch
from safetensors import safe_open
from safetensors.torch import save_file


# File paths
model1_path = "nsfw.safetensors"
model2_path = "flux1-dev-fp8.safetensors"
output_path = "outut.safetensors"

# Mixing ratio
alpha = 0.3
beta = 1 - alpha  # This will be 0.60

print(f"Loading models from:\nModel 1: {model1_path}\nModel 2: {model2_path}")

# Load model1
model1 = {}
with safe_open(model1_path, framework='pt') as f:
    for key in f.keys():
        model1[key] = f.get_tensor(key)
print("Model 1 loaded successfully.")

# Load model2
model2 = {}
with safe_open(model2_path, framework='pt') as f:
    for key in f.keys():
        model2[key] = f.get_tensor(key)
print("Model 2 loaded successfully.")

# Merge the models according to the specified ratio
print(f"Merging models with ratio:\nModel 1 (alpha): {alpha}\nModel 2 (beta): {beta}")
merged_model = {}
for key in model1.keys():
    if key in model2:
        merged_model[key] = model1[key] * alpha + model2[key] * beta
        print(f"Merged tensor for key: {key}")
    else:
        merged_model[key] = model1[key]
        print(f"Key {key} only found in Model 1, using its tensor.")

# For keys that are in model2 but not in model1
for key in model2.keys():
    if key not in model1:
        merged_model[key] = model2[key]
        print(f"Key {key} only found in Model 2, using its tensor.")

# Save the merged model as SafeTensors
save_file(merged_model, output_path)
print(f"Merged model saved as {output_path}")
