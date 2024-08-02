from safetensors.torch import load_file

def load_model(filepath):
    return load_file(filepath)

def list_model_keys(filepath):
    model = load_model(filepath)
    print(f"Keys in model {filepath}:")
    for key in model.keys():
        print(key)
    print()

if __name__ == "__main__":
    model1_path = "safetensors-merge-supermario/sd_xl_turbo_1.0.safetensors"
    model2_path = "safetensors-merge-supermario/aura_flow_0.2.safetensors"

    list_model_keys(model1_path)
    list_model_keys(model2_path)
