import os

def create_file_in_folder(path, filename):
    # Check if the folder exists
    if not os.path.isdir(path):
        print(f"Folder '{path}' does not exist. Creating folder.")
        os.makedirs(path)
    
    # Create the file path
    file_path = os.path.join(path, filename)
    
    # Check if the file already exists
    if not os.path.isfile(file_path):
        # Create and write to the file
        with open(file_path, 'w') as file:
            file.write("This is a new file.")
        print(f"File '{filename}' created in folder '{path}'.")
    else:
        print(f"File '{filename}' already exists in folder '{path}'.")

# Example usage
path = "save2"
filename = "example.txt"
create_file_in_folder(path, filename)
