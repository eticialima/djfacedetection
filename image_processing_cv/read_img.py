import numpy as np
import cv2

# read image
img = cv2.imread('./images/car_001.jpg')
print(img)
print(img.max)
print(img.min)
print(img.shape)
