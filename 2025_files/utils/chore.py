import os

def rename_files_in_folder(folder_path, prefix="", suffix="", new_name=None):
    """
    Rename files in a folder with an optional prefix, suffix, or completely new name.

    Args:
        folder_path (str): Path to the folder containing the files.
        prefix (str): Optional prefix to add to the file names.
        suffix (str): Optional suffix to add to the file names.
        new_name (str): If provided, all files will be renamed to this base name with an index appended.
                        (e.g., 'file_1.txt', 'file_2.txt')

    Returns:
        None
    """
    try:
        # Get a list of all files in the folder
        files = os.listdir(folder_path)
        
        for index, file_name in enumerate(files):
            # Get the full path of the original file
            old_file_path = os.path.join(folder_path, file_name)
            
            # Skip directories, process only files
            if not os.path.isfile(old_file_path):
                continue
            
            # Get file extension
            file_extension = os.path.splitext(file_name)[1]
            
            # Determine the new file name
            if new_name:
                # Rename all files to the new name with index
                new_file_name = f"{new_name}_{index + 1}{file_extension}"
            else:
                # Add prefix and suffix to the original file name
                base_name = os.path.splitext(file_name)[0]
                new_file_name = f"{prefix}{base_name}{suffix}{file_extension}"
            
            # Get the full path of the new file
            new_file_path = os.path.join(folder_path, new_file_name)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} -> {new_file_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    # Path to the folder containing files
    folder_path = "./example_folder"

    # Rename files by adding a prefix and suffix
    rename_files_in_folder(folder_path, prefix="new_", suffix="_edited")

    # Alternatively, rename all files to a new base name
    # rename_files_in_folder(folder_path, new_name="renamed_file")