import torch
from safetensors import safe_open

# Path to your SafeTensors file
file_path = "pytorch_lora_weights.safetensors"

# Open the SafeTensors file using PyTorch
with safe_open(file_path, framework="torch") as f:
    # List all the tensors stored in the file
    tensor_names = f.keys()
    print(f"Tensors available: {tensor_names}")
    
    # Extract specific tensors
    for name in tensor_names:
        tensor = f.get_tensor(name)
        print(f"Tensor '{name}' shape: {tensor.shape}")
        print(f"Tensor '{name}' dtype: {tensor.dtype}")
        # Now you can use the tensor in PyTorch
