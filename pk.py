import torch
from safetensors import safe_open
from safetensors.torch import save_file

def merge_models_in_chunks(model1_path, model2_path, output_path, alpha=0.4):
    with safe_open(model1_path, framework="pt") as model1, safe_open(model2_path, framework="pt") as model2:
        common_keys = set(model1.keys()).intersection(set(model2.keys()))

        merged_model = {}
        for key in common_keys:
            # Load tensors for the current key from both models
            print(key)
            tensor1 = model1.get_tensor(key)
            tensor2 = model2.get_tensor(key)

            # Ensure tensors are on the same device and dtype for the operation
            dtype = tensor1.dtype
            tensor1 = tensor1.to(torch.float32)
            tensor2 = tensor2.to(torch.float32)

            # Perform the merging operation
            mixed_tensor = alpha * tensor1 + (1 - alpha) * tensor2

            # Convert back to the original dtype and store in the merged model dictionary
            merged_model[key] = mixed_tensor.to(dtype)

            # Save the merged chunk immediately to avoid excessive memory use
            save_file(merged_model, output_path)

            # Clear the merged_model dictionary to free memory
            merged_model.clear()

    print(f"Merged model saved to {output_path}")

if __name__ == "__main__":
    model1_path = "modal.safetensors"
    model2_path = "model.safetensors.1"
    output_path = "kaoutput.safetensors"
    alpha = 0.4  # Mixing ratio

    # Merge models in a memory-efficient manner
    merge_models_in_chunks(model1_path, model2_path, output_path, alpha)
