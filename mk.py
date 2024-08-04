from safetensors.torch import load_file, save_file
import torch

def load_model(filepath):
    return load_file(filepath)

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
    model1_path = "safetensors-merge-supermario/flux1-schnell-fp8.safetensors"
    model2_path = "safetensors-merge-supermario/flux1-dev-fp8.safetensors"
    output_path = "safetensors-merge-supermario/mixed_model.safetensors"
    alpha = 0.5  # Mixing ratio

    model1 = load_model(model1_path)
    model2 = load_model(model2_path)

    mixed_model = mix_models(model1, model2, alpha)
    print(mixed_model)

    save_model(mixed_model, output_path)

    print(f"Mixed model saved to {output_path}")
