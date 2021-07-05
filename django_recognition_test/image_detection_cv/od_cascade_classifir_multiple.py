import cv2
import numpy as np

# face detection multiple
img = cv2.imread('./images/friends.jpg')

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# load cascade classifir - frontalface
face_cascade = cv2.CascadeClassifier('./cascade_classifier/haarcascade_frontalface_default.xml')

# apply cascade classifir to an image  
faces,num_detection = face_cascade.detectMultiScale2(img, minNeighbors=8)

# run a loop
for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
# display the image
cv2.imshow('face detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(faces)
print(num_detection)
 
