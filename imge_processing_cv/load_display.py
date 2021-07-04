import numpy as np
import cv2
import matplotlib.pyplot as plt

# read image
img = cv2.imread('./images/car_001.jpg')

# show image
# cv2.imshow('Origial image', img) 
# cv2.waitKey(10000)  # this command will the windows for milliseconds
# cv2.destroyAllWindows()

# cv2.waitKey(0) # this command display the windows untill you press x or any keyboard key


# img RGB
plt.imshow(img[:,:,[2,1,0]]) # assume that image is in RGB format
plt.show()
 