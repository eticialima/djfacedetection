import numpy as np
import cv2
import pandas as pd
import os 
import pickle

# consider sample image
img = cv2.imread('./images/Sachin Tendulkar/2200.jpg')

# cv2.imshow('sample', image) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    
face_detection_model = './models/res10_300x300_ssd_iter_140000.caffemodel'
face_detection_proto = './models/deploy.prototxt.txt'
face_descriptor = './models/openface.nn4.small2.v1.t7'

# load models using cv2 dnn
detector_model = cv2.dnn.readNetFromCaffe(face_detection_proto,face_detection_model)
descriptor_model = cv2.dnn.readNetFromTorch(face_descriptor)


def helper(image_path):
    img = cv2.imread(image_path)
    #1 face detection
    image = img.copy()
    h,w = image.shape[:2]
    img_blob = cv2.dnn.blobFromImage(image,1,(300,300),(104,177,123),swapRB=False,crop=False)
    #set the input 
    detector_model.setInput(img_blob)
    detections = detector_model.forward()

    if len(detections > 0):
        i = np.argmax(detections[0,0,:,2]) # consider the face with max confidence score
        confidence = detections[0,0,i,2]
        if confidence > 0.5:
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startx, starty, endx, endy) = box.astype('int')
            #2 feature extraction or Embeddiong
            roi = image[starty:endy,startx:endx].copy()
            #get the face descriptors
            faceblob = cv2.dnn.blobFromImage(roi,1/255,(96,96),(0,0,0,),swapRB=True,crop=True) 
            descriptor_model.setInput(faceblob)
            vectors = descriptor_model.forward()
           
            return vectors
    return None 
            
folders = os.listdir('images')
for folder in folders:
    filenames = os.listdir('images/Aamir Khan')
    for filename in filenames:
        try: 
            vector = helper('./images/{}{}'.format(folder, filename))
            if vector is not None:
                data['data'].append(vector)
                data['label'].append(filename)
                print('feature extraction sucessfully')
        except:
            pass
        
 