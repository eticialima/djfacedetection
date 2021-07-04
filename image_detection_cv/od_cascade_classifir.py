import cv2
import numpy as np

# face detection

img = cv2.imread('./images/0021.jpg')

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# load cascade classifir - frontalface
face_cascade = cv2.CascadeClassifier('./cascade_classifier/haarcascade_frontalface_default.xml')

# apply cascade classifir to an image  
faces,num_detection = face_cascade.detectMultiScale2(img)

print(faces)
print(num_detection)

#draw the rectangle
pt1 = (114,40)
pt2 = (114+196, 40+196)
cv2.rectangle(img, pt1, pt2, (0,0,255))

#draw the Circle
cx = 114 + 192//2
cy = 40 + 192//2
r = 196//2 
cv2.circle(img, (cx,cy), r, (0,255,255),2)

cv2.imshow('face detection circle',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
