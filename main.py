import os
import sys

# Define paths
image_folder = 'dataset/images/'
# List all files in the image folder
image_files = os.listdir(image_folder)

def execute_component_matching(image_path):
    # Import and execute component_matching.py
    import component_matching
    component_matching.main(image_path)

if __name__ == "__main__":
    # Camera calibration
    # Not possible with these images

    # Define with true which method to execute
    execute_component_matching_flag = True 

    # Determine which scripts to execute based on conditions
    if execute_component_matching_flag:
        # image_path = 'dataset\images\G34_bottom_pp (2).jpg'
        # image_path = 'dataset\images\G29_bottom_pp (2).jpg'
        # execute_component_matching(image_path)

        # Loop through each image file in the folder
        for image_file in image_files:
            if image_file.endswith(".jpg"):  # Ensure it's an image file
                image_path = os.path.join(image_folder, image_file)
                execute_component_matching(image_path)
        
    # Other methods
    # if YOLO_flag: