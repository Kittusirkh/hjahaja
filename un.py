import torch
from safetensors import safe_open
from safetensors.torch import save_file

def load_model(filepath, device='cpu'):
    model = {}
    with safe_open(filepath, framework="pt", device=device) as f:
        for key in f.keys():
            model[key] = f.get_tensor(key)
    return model

def save_model(model, filepath):
    save_file(model, filepath)

def convert_tensor(tensor, dtype=torch.float32):
    return tensor.to(dtype)

def mix_models(model1, model2, alpha=0.5):
    mixed_model = {}
    common_keys = set(model1.keys()).intersection(set(model2.keys()))
    for key in common_keys:
        tensor1 = convert_tensor(model1[key])
        tensor2 = convert_tensor(model2[key])
        mixed_tensor = alpha * tensor1 + (1 - alpha) * tensor2
        mixed_model[key] = mixed_tensor.to(model1[key].dtype)  # Convert back to original dtype if necessary
    return mixed_model

if __name__ == "__main__":
    model1_path = "safetensors-merge-supermario/model.safetensors"
    model2_path = "safetensors-merge-supermario/mixed_model.safetensors"
    output_path = "safetensors-merge-supermario/output.safetensors"
    alpha = 0.3  # Mixing ratio

    # Load models in a memory-efficient manner
    model1 = load_model(model1_path)
    model2 = load_model(model2_path)

    # Mix models
    mixed_model = mix_models(model1, model2, alpha)

    # Print the mixed model for debugging
    print(mixed_model)

    # Save the mixed model
    save_model(mixed_model, output_path)

    print(f"Mixed model saved to {output_path}")
