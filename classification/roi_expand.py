import cv2
import os
import numpy as np

image_folder = 'data/masks' 
image_filenames = os.listdir(image_folder)

for i, image_filename in enumerate(image_filenames):
    image_path = os.path.join(image_folder, image_filename)
    
    roi_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, thresholded = cv2.threshold(roi_image, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_result = np.zeros_like(roi_image)

    cv2.drawContours(contours_result, contours, -1, (255), thickness=cv2.FILLED)
 
    # Define a vertical kernel
    kernel = np.ones((30, 1), np.uint8)  # Change the kernel size to adjust the extension length
    dilated_result = cv2.dilate(contours_result, kernel, iterations=1)
    cv2.imwrite('masks/'+image_filename, dilated_result)
