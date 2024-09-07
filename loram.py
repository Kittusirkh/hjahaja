import torch
from diffusers import FluxPipeline

# Load the Mystic model
pipe = FluxPipeline.from_pretrained("enhanceateam/mystic", torch_dtype=torch.float16)

# Load and merge the LoRA weights into the model
pipe.load_lora_weights("TheLastBen/The_Hound", weight_name="sandor_clegane_single_layer.safetensors")

# Optionally, merge the LoRA with the base model
for name, module in pipe.named_modules():
    if hasattr(module, "merge_lora_weights"):
        module.merge_lora_weights()

# Get the state dict of the model
state_dict = pipe.state_dict()

# Save the merged model as a safetensors file
from safetensors.torch import save_file
save_file(state_dict, "path_to_save_model/merged_mystic_model.safetensors")
