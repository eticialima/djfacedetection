import numpy as np
import cv2 

# read image
img = cv2.imread('./images/3_cars.jpg')

# cv2.imshow('original_img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(img.shape)

b,g,r = cv2.split(img)
# print(b)

b.shape,g.shape,r.shape,

# cv2.imshow('blue image', b)
# cv2.imshow('green image', g)
# cv2.imshow('red image', r)
# cv2.imshow('original image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
 
