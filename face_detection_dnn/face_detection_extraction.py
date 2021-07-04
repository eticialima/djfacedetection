import cv2
import numpy as np
import dlib 
from imutils import face_utils

# read image
img = cv2.imread('./images/girl.png')
 
# shape predictor
shape_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
# face_descriptor
shape_descriptor = dlib.face_recognition_model_v1('./models/dlib_face_recognition_resnet_model_v1.dat')

# step-1: Face detection
image = img.copy()
face_detector = dlib.get_frontal_face_detector()

faces = face_detector(image)
for box in faces:
    pt1 = box.left(), box.top()
    pt2 = box.right(), box.bottom()
    
    face_shape = shape_predictor(image,box)
    face_shape_array = face_utils.shape_to_np(face_shape)
    
    # shape_descriptor.shape_descriptor.compute_face_descriptor(image,face_shape)
   
    # print(face_shape_array)
    for point in face_shape_array:
        cv2.circle(image, tuple(point),3,(255,255,0),-1)
    
    cv2.rectangle(image, pt1, pt2,(0,0,255))
    
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()