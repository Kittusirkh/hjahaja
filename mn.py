import numpy as np
import torch
import safetensors.torch

def load_model(filepath):
    return safetensors.torch.load_file(filepath)

def save_model(model, filepath):
    safetensors.torch.save_file(model, filepath)

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

    # Memory-map the tensors to avoid loading entire files into RAM
    model1 = torch.load(model1_path, map_location=torch.device('cpu'))
    model2 = torch.load(model2_path, map_location=torch.device('cpu'))

    mixed_model = mix_models(model1, model2, alpha)
    print(mixed_model)

    save_model(mixed_model, output_path)

    print(f"Mixed model saved to {output_path}")
