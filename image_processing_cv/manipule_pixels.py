import numpy as np
import cv2
import matplotlib.pyplot as plt

# read image
img = cv2.imread('./images/car_001.jpg')
 
print(img.shape)

def display(winame, image):
    cv2.imshow(winame, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
display('cars', img)
# type(img)
  
# acess first 300 rows (height) and first (wdth) of the image (numpy array)
# roi = img[0:300, 0:300]
# display('acess', roi)

# box green in area roi 
# green_pixels = (0, 255, 0)
# roi = img[0:300, 0:300] = green_pixels 
# display('manipulate', img)