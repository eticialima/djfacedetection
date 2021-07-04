import cv2
import numpy as np
import dlib 
from imutils import face_utils

# read image
img = cv2.imread('./images/girl.png')

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# load the model
net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt',
                               './models/res10_300x300_ssd_iter_140000_fp16.caffemodel')

# extract blob
blob = cv2.dnn.blobFromImage(img,1,(300,300),(104,177,123),swapRB=False)

# the blob as input 
net.setInput(blob) 
# run the model
detections = net.forward()

detections.shape

h,w = img.shape[:2]
for i in range(0,detections.shape[2]):
    confidence = detections[0,0,i,2] 
    if confidence >= 0.5:
        # print(confidence)
        # bounding box (3:7)
        box = detections[0,0,i,3:7]
        box = box*np.array([w,h,w,h])
        box = box.astype(int)
        startx, starty, endx, endy = box
        
        # draw the rectangle
        cv2.rectangle(img, (startx,starty),(endx,endy),(0,255,0))
        
        # put text 
        text = 'Face: {:.2f} %'.format(confidence*100)
        cv2.putText(img, text,(startx,starty-10),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255))
        
        
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()