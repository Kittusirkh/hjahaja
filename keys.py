import os
import torch
from safetensors import safe_open

def check_keys_in_file(filename, log_filename):
    # Create the file paths
    file_path = filename
    log_path = log_filename
    
    # Check if the file exists
    if not os.path.isfile(file_path):
        with open(log_path, 'a') as log_file:
            log_file.write(f"{filename} does not exist.\n")
        print(f"{filename} does not exist.")
        return
    
    # Check for keys in the file
    try:
        with safe_open(file_path, framework="pt") as f:
            keys = f.keys()
            with open(log_path, 'a') as log_file:
                log_file.write(f"Keys in file '{filename}': {keys}\n")
            print(f"Keys in file '{filename}': {keys}")
    except Exception as e:
        with open(log_path, 'a') as log_file:
            log_file.write(f"Error opening or reading '{filename}': {e}\n")
        print(f"Error opening or reading '{filename}': {e}")

# Example usage
filename = "sd_xl_turbo_1.0.safetensors"
log_filename = "logsdxl.txt"

check_keys_in_file(filename, log_filename)
