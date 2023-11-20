import os

def create_relative_directory(directory_path):
    current_directory = os.getcwd()
    new_directory = os.path.join(current_directory, directory_path)
    
    # Check if the directory already exists
    if not os.path.exists(new_directory):
        try:
            # Create the directory if it doesn't exist
            os.makedirs(new_directory)
            print(f"Directory '{directory_path}' created successfully.")
        except OSError as err:
            print(f"Error: {err}")
    else:
        print(f"Directory '{directory_path}' already exists.")
    
    return new_directory

