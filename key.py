import os
import torch
import safetensors
from safetensors import safe_open

def check_key_in_file(path, filename, key, log_filename):
    # Create the file paths
    folder_path = os.path.join(path)
    file_path = os.path.join(folder_path, filename)
    log_path = os.path.join(folder_path, log_filename)
    
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    
    # Check if the file exists
    if not os.path.isfile(file_path):
        with open(log_path, 'a') as log_file:
            log_file.write(f"{filename} does not exist in {folder_path}.\n")
        print(f"{filename} does not exist in {folder_path}.")
        return
    
    # Check for the key in the file
    try:
        with safe_open(file_path, framework="pt") as f:
            if key in f.keys():
                with open(log_path, 'a') as log_file:
                    log_file.write(f"Key '{key}' found in file '{filename}'.\n")
                print(f"Key '{key}' found in file '{filename}'.")
            else:
                with open(log_path, 'a') as log_file:
                    log_file.write(f"Key '{key}' not found in file '{filename}'.\n")
                print(f"Key '{key}' not found in file '{filename}'.")
    except Exception as e:
        with open(log_path, 'a') as log_file:
            log_file.write(f"Error opening or reading '{filename}': {e}\n")
        print(f"Error opening or reading '{filename}': {e}")

# Example usage
path = "save2"
filename = "flux1-schnell.sft"
key = "your_key_here"
log_filename = "log.txt"

check_key_in_file(path, filename, key, log_filename)
