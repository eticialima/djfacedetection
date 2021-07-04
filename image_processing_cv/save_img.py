import numpy as np
import cv2
import matplotlib.pyplot as plt

# read image
img = cv2.imread('./images/car_001.jpg')

# save image
img = cv2.imwrite('./images/save.png', img)

 