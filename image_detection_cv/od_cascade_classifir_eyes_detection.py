import cv2
import numpy as np

# face detection multiple
img = cv2.imread('./images/friends.jpg')

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# load cascade classifir - frontalface
face_cascade = cv2.CascadeClassifier('./cascade_classifier/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./cascade_classifier/haarcascade_eye.xml')

#1 apply cascade classifir to an image  
faces,num_detection_face = face_cascade.detectMultiScale2(img, minNeighbors=8)

#2 run a loop
for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255))

    #3 eye detection
    face_roi = img[y:y+h,x:x+h] # croping the image
    # apply to cascade classifir (eye)
    eyes, num_detection_eyes = eye_cascade.detectMultiScale2(face_roi)
    # run a loop
    for ex, ey, ew, eh in eyes:
        cx = x+ex+ew//2
        cy = y+ey+eh//2
        r = eh//2 
        cv2.circle(img,(cx,cy),r,(0,255,255))
        
# display the image
cv2.imshow('face eye detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

 