"""
Muhammad Ahsan Baig's Code
"""
import cv2

img = cv2.imread('rose.jpg')

print("Image Properties")
print("- Number of Pixels: " + str(img.size))
print("- Shape/Dimensions: " + str(img.shape))