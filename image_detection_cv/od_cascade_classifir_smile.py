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
smile_cascade = cv2.CascadeClassifier('./cascade_classifier/haarcascade_smile.xml')

# FACE cascade classifir
faces,num_detection_face = face_cascade.detectMultiScale2(img, minNeighbors=8)
for x,y,w,h in faces:
    face_roi = img[y:y+h,x:x+h].copy() # croping the image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
 
    # EYE cascade classifir (eye)
    eyes, num_detection_eyes = eye_cascade.detectMultiScale2(face_roi)
    for ex, ey, ew, eh in eyes:
        cx = x+ex+ew//2
        cy = y+ey+eh//2
        r = eh//2 
        cv2.circle(img,(cx,cy),r,(0,255,0),2)
        
    # SMILE cascade classifir
    smile, num_detection_smiles = smile_cascade.detectMultiScale2(face_roi)
    for sx, sy, sw,sh in smile:
        cv2.rectangle(img,(x+sx,y+sy),(x+sx+sw, y+sy+sh),(0,0,255),2)
        
        
# display the image
cv2.imshow('face eye detected smile', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 