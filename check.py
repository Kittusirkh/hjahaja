import os

def check_path_folder_and_file(path):
    # Check if the folder exists
    if os.path.isdir(path):
        print(f"Folder '{path}' exists.")
        
        # List all files in the folder
        files = os.listdir(path)
        if files:
            print(f"Files in the folder '{path}':")
            for file in files:
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path):
                    print(f"  - {file}")
                else:
                    print(f"  - {file} (not a file)")
        else:
            print(f"No files found in the folder '{path}'.")
    else:
        print(f"Folder '{path}' does not exist.")

# Example usage
path = "save2"
check_path_folder_and_file(path)
