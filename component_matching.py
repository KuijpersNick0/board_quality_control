import cv2
import numpy as np
import os 
import tkinter as tk
from tkinter import messagebox

# Display alert
def show_alert():
    messagebox.showerror("Alert", "Components not correctly placed!")

# Create a GUI window 
root = tk.Tk()
root.withdraw()  # Hide the main window

def segment_image(img):
    # keep only the 2 largest components 
    # I first tried this method :
    # Apply watershed algorithm to segment the image : https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
    # But I realised that it was not the best method for this task, as it will not be able to segment the components
    # Thus used simple segmentation with thresholding and morphological operations
    # New problem : threshold with images where one component is in dark area
    # Thus used adaptive thresholding
    
    # I receive grayscale image, convert to BGR for last display
    gray = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Threshold the grayscale image to obtain a binary mask
    # Problem with this method : threshold with images where one component is in dark area
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Adaptive thresholding better for dark images
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 61, -2)

    # Apply morphological operations (erosion and dilation) to remove unwanted connections, etc.
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # cv2.namedWindow("Display new window 3", cv2.WINDOW_NORMAL) 
    # cv2.imshow("Display new window 3", opening)
    # cv2.waitKey(0)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Extract the two largest contours (assuming they represent the components)
    if len(contours) >= 2:
        component1 = contours[0]
        component2 = contours[1]

        # Create masks for the two components
        mask1 = np.zeros_like(gray)
        mask2 = np.zeros_like(gray)
        cv2.drawContours(mask1, [component1], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(mask2, [component2], -1, 255, thickness=cv2.FILLED)

        # Combine the masks to get the segmented components
        segmented_components = cv2.bitwise_or(mask1, mask2)

        # Apply the segmented mask to the original image
        segmented_image = cv2.bitwise_and(img, img, mask=segmented_components)
    
        # Display the segmented image
        cv2.namedWindow("Display new window", cv2.WINDOW_NORMAL) 
        cv2.imshow("Display new window", segmented_image)
        cv2.waitKey(0)

        return segmented_image
    
    else:
        print("Could not segment two components")
        return None

def calculate_rotation(segmentedImage):
    # Calculate rotation components with findContours : https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    # Ensure that segmentedImage is a binary (CV_8UC1) image
    if len(segmentedImage.shape) == 3:
        # Convert to grayscale if it's a color image
        gray_image = cv2.cvtColor(segmentedImage, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = segmentedImage
    
    # Threshold the grayscale image to obtain a binary image
    # _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # cv2.namedWindow("Display new window 2", cv2.WINDOW_NORMAL) 
    # cv2.imshow("Display new window 2", binary_image)
    # cv2.waitKey(0)

    # We need to return 2 rotations, one for each component
    rotations = []
    # Find contours on the binary image
    contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if(contours):
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:2]
    
        for contour in contours:
            # Understand rotation values : https://theailearner.com/tag/cv2-minarearect/
            rect = cv2.minAreaRect(contour) 
            rotation = rect[2]
            # box = cv2.boxPoints(rect)
            # box = np.int0(box) 
            # cv2.drawContours(gray_image, [box], 0, (255,0,0), 3)
            # cv2.namedWindow("Display new window 3", cv2.WINDOW_NORMAL) 
            # cv2.imshow("Display new window 3", gray_image)
            # cv2.waitKey(0)
            rotations.append(rotation)
    return rotations

def check_component(image):
    # Segment the image
    segmentedImage = segment_image(image)
    # Calculate rotation components
    rotation = calculate_rotation(segmentedImage)
    print(rotation)
    
    correctly_placed = [False, False]

    for i, rotation_elem in enumerate(rotation):
        # Allow 3 degrees of rotation around axis
        if (0 <= rotation_elem <= 3) or (87 <= rotation_elem <= 90):
            correctly_placed[i] = True
    if all(correctly_placed):
        return "Components correctly placed"
    else :
        return "Component misplaced"

def find_match(image, template, threshold=0.9):
    # Apply normalized cross-correlation between the ROI and the template
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    matchLoc = max_loc    
    # img_display = image.copy()

    if max_val >= threshold:
        # cv2.rectangle(img_display, matchLoc, (matchLoc[0] + template.shape[0], matchLoc[1] + template.shape[1]), (0,0,0), 2, 8, 0 ) 
        img_component = image[matchLoc[1]:matchLoc[1] + template.shape[0], matchLoc[0]:matchLoc[0] + template.shape[1]] 
        cv2.namedWindow("Display window", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Display result", cv2.WINDOW_NORMAL)

        cv2.imshow("Display window", image)
        cv2.waitKey(0)
        cv2.imshow("Display result", img_component)
        cv2.waitKey(0)
        print("Components matching found")
        return check_component(img_component) 
    else:
        return "Components matching not found"

def main(image_path):
    # The first thing I tried to do is to use cv2.matchTemplate() on identified ROI of the image
    # But I stumbled on multiple problems:
    # - The ROI's are not always located at the same place, the board is not photographed from the same angle/position
    # - matchTemplate will find the component in the ROI, but it will not be able to tell if it is misplaced or not
    # - matchTemplate will not be able to tell if one component is missing, etc...
    # I thus came with a second method, which is to use the ROI's to find the components in the image
    # Then, I will be able to tell if the components are misplaced or not, if one is missing, etc...

    # Loading image  
    image = cv2.imread(image_path) 
    # Filter
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Contour detection with Canny from cv2 to detect board and align with this informations roi's
    # 
    # ! Realised later on that this method will not work, some images do not contain the whole board
    # 
    # edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)

    # ROI of components extracted from roi_extract.py
    # double units
    # [[3765 3051  205  277]
    #  [4009 3051  205  277]]

    # single unit
    # [3714 3051  538  314]
    # x1, y1 = 3765, 3051
    # x2, y2 = 4009, 3051
    # w, h = 205, 277
    # roi_component1 = gray_image[y1:y1+h, x1:x1+w]
    # roi_component2 = gray_image[y2:y2+h, x2:x2+w]  

    # cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("Display", roi_component1)
    # cv2.waitKey(0)
    # cv2.imshow("Display", roi_component2)
    # cv2.waitKey(0)  
 
    # Load template images for the two components
    # Construct paths for template images
    template_folder = 'dataset/images/roi_images'
    # template_path1 = os.path.join(template_folder, '0_roi.jpg')
    # template_path2 = os.path.join(template_folder, '1_roi.jpg')
    template_path3 = os.path.join(template_folder, '3_roi.jpg')

    # template_component1 = cv2.imread(template_path1, cv2.IMREAD_GRAYSCALE)
    # template_component2 = cv2.imread(template_path2, cv2.IMREAD_GRAYSCALE)
    template_component3 = cv2.imread(template_path3, cv2.IMREAD_GRAYSCALE)

    # First : Check if components are misplaced
    # Now find component in image
    # misplaced1 = find_match(gray_image, template_component1)
    # misplaced2 = find_match(gray_image, template_component2)
    found_match = find_match(gray_image, template_component3)

    # print(misplaced1)
    # print(misplaced2)
    print(found_match)

    if(found_match == "Component misplaced"):
        show_alert()


if __name__ == "__main__":
    # image_path = 'dataset/images/G42_bottom_pp (2).jpg'
    image_path = 'dataset/images/G33_bottom_pp (1).jpg'
    main(image_path)
 
    # root.mainloop()