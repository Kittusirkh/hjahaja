from huggingface_hub import hf_hub_upload

# Replace 'pranavajay/Kalpana' with your repo_id
repo_id = 'pranavajay/Kalpana'

# Path to your file
file_path = 'model.safetensors'

# Upload the file
hf_hub_upload(repo_id=repo_id, path_or_fileobj=file_path, filename='model.safetensors')
