import numpy as np
import cv2 

# read image
img = cv2.imread('./images/beach.jpg') # bgr

# cv2.imshow('original_img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# BGR to gray 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# BGR to RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# BGR to Lab
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

cv2.imshow('original-bgr', img)
cv2.imshow('gray', gray)
cv2.imshow('rgb', rgb)
cv2.imshow('hsv', hsv)
cv2.imshow('lab', lab)


cv2.waitKey(0)
cv2.destroyAllWindows()