from safetensors.torch import load_file, save_file
import torch

def load_model(filepath):
    return load_file(filepath)

def save_model(model, filepath):
    save_file(model, filepath)

def mix_tensors(tensor1, tensor2, alpha):
    return alpha * tensor1 + (1 - alpha) * tensor2

def mix_models(model1, model2, alpha=0.5, additional_keys=None, mix_ratios=None):
    mixed_model = {}

    # Determine the common keys
    common_keys = set(model1.keys()).intersection(set(model2.keys()))
    unique_keys_model1 = set(model1.keys()) - common_keys
    unique_keys_model2 = set(model2.keys()) - common_keys

    # Mix common keys
    for key in common_keys:
        if mix_ratios and key in mix_ratios:
            mixed_model[key] = mix_tensors(model1[key], model2[key], mix_ratios[key])
        else:
            mixed_model[key] = mix_tensors(model1[key], model2[key], alpha)

    # Handle unique keys from model1
    for key in unique_keys_model1:
        mixed_model[key] = model1[key]

    # Handle unique keys from model2
    for key in unique_keys_model2:
        mixed_model[key] = model2[key]

    # Add additional keys if provided
    if additional_keys:
        for key, value in additional_keys.items():
            mixed_model[key] = value

    return mixed_model

if __name__ == "__main__":
    model1_path = "safetensors-merge-supermario/sd_xl_turbo_1.0.safetensors"
    model2_path = "safetensors-merge-supermario/aura_flow_0.2.safetensors"
    output_path = "safetensors-merge-supermario/mix_model.safetensors"
    alpha = 0.5  # Default mixing ratio

    # Example of specific mixing ratios for certain keys
    specific_mix_ratios = {
        "key1": 0.7,
        "key2": 0.3
    }

    # Example of additional keys to be added to the mixed model
    additional_keys = {
        "new_key": torch.randn(5, 5)  # Example tensor
    }

    model1 = load_model(model1_path)
    model2 = load_model(model2_path)

    mixed_model = mix_models(model1, model2, alpha, additional_keys, specific_mix_ratios)

    save_model(mixed_model, output_path)

    print(f"Mixed model saved to {output_path}")
