"""
Muhammad Ahsan Baig's Code
"""
from google.colab.patches import cv2_imshow

blue, green, red = cv2.split(img) # Split the image into its channels
img_gs = cv2.imread('rose.jpg', cv2.IMREAD_GRAYSCALE) # Convert image to grayscale

cv2_imshow(red) # Display the red channel in the image
cv2_imshow(blue) # Display the red channel in the image
cv2_imshow(green) # Display the red channel in the image
cv2_imshow(img_gs) # Display the grayscale version of image