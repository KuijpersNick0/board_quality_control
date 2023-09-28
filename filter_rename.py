import os
import shutil

# Function to rename and move images
def rename_and_move_images(src_dir, dest_dir):
    for root, _, files in os.walk(src_dir):
        for file in files:
            # if file.startswith("pp (1)") or file.startswith("pp (2)") or file.startswith("PP (1)") or file.startswith("PP (2)"):
            if "pp (1)" in file or "pp (2)" in file or "PP (1)" in file or "PP (2)" in file:
                folder_name = os.path.basename(os.path.dirname(root)) 
                position = "Bottom" if "bottom" in root.lower() else "Top"
                extension = os.path.splitext(file)[-1].lower()
                new_name = f"{folder_name}_{position}_PP{file.split('(')[-1].split(')')[0]}{extension}"
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, new_name)
                shutil.copy(src_path, dest_path)
                print(f"Renamed: {file} -> {new_name}")

# Source and destination directories
src_directory = "C:/Users/shop/Documents/RailNova/data"
dest_directory = "C:/Users/shop/Documents/RailNova/data_processed"

# Create the destination directory if it doesn't exist
os.makedirs(dest_directory, exist_ok=True)

# Call the function to rename and move the images
rename_and_move_images(src_directory, dest_directory)