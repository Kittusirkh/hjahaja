from safetensors.torch import load_file, save_file

def load_model(filepath):
    return load_file(filepath)

def save_model(model, filepath):
    save_file(model, filepath)

def mix_models(model1, model2, alpha=0.5):
    # Assuming models are dictionaries of tensors
    mixed_model = {}
    common_keys = set(model1.keys()).intersection(set(model2.keys()))
    for key in common_keys:
        mixed_model[key] = alpha * model1[key] + (1 - alpha) * model2[key]
    return mixed_model

if __name__ == "__main__":
    model1_path = "safetensors-merge-supermario/sd_xl_turbo_1.0.safetensors"
    model2_path = "safetensors-merge-supermario/aura_flow_0.2.safetensors"
    output_path = "safetensors-merge-supermario/mixed_model.safetensors"
    alpha = 0.5  # Mixing ratio

    model1 = load_model(model1_path)
    model2 = load_model(model2_path)

    mixed_model = mix_models(model1, model2, alpha)

    save_model(mixed_model, output_path)

    print(f"Mixed model saved to {output_path}")
