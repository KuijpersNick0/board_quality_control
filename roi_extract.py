import cv2
import numpy as np
import os

# image_path
img_path="dataset\images\G29_bottom_pp (2).jpg"


# read image
img_raw = cv2.imread(img_path)
img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
# Adjust the dimensions as needed
# resized_image = cv2.resize(img_raw, (800, 600))  

# select ROIs function
ROIs = cv2.selectROIs("Display", img_raw, True)

# print rectangle points of selected roi
print(ROIs)

#counter to save image with different name
crop_number=3 

#loop over every bounding box save in array "ROIs"
for rect in ROIs:
    x1=rect[0]
    y1=rect[1]
    x2=rect[2]
    y2=rect[3]

    #crop roi from original image
    img_crop = img_raw[y1:y1+y2,x1:x1+x2] 

    #show cropped image
    cv2.imshow("crop"+str(crop_number),img_crop)

    #save cropped image 
    output_folder = 'dataset/images/roi_images'
    output_path = os.path.join(output_folder, str(crop_number) + '_roi.jpg')
    cv2.imwrite(output_path , img_crop)
        
    crop_number += 1

#hold window
cv2.waitKey(0)