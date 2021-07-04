import cv2 
import numpy as np 

# read image original
img = cv2.imread('./images/car_001.jpg')

# canvas = np.zeros((300,300,3),dtype='uint8')

# cv2.imshow('canvas', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# left to right
# pt1 = (0,0)
# pt2 = (300, 300)
# color = (0,255,0)
# cv2.line(canvas, pt1, pt2, color)

# cv2.imshow('canvas', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# right to left
# pt1 = (0,300)
# pt2 = (300, 0) 
# cv2.line(canvas, pt1, pt2, (0,255,0),5)

# cv2.imshow('canvas', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

### Rectangle ### 

# canvas = np.zeros((600,600,3),dtype='uint8')

# pt1 = (50,50)
# pt2 = (100, 100)
# color = (255,0,0)

# cv2.rectangle(canvas, pt1, pt2, color)
# cv2.rectangle(canvas, (100,100),(200,200),(0,255,0),5)
# cv2.rectangle(canvas, (200,200),(300,300),(0,0,255),-1)

# cv2.imshow('canvas', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

### Polygon ### 

# canvas = 255*np.ones((600,600,3),dtype='uint8')

# # points of hexagon
# pts = np.array([[200,200],[250,300],[350,300],[400,200],[350,100],[250,100]])
# pts = pts.reshape(-1,1,2) 
# cv2.polylines(canvas, [pts], True,(255,0,0),5)
 
# pts = np.array([[100,200],[150,300],[250,300],[300,200],[250,100],[150,100]])
# pts = pts.reshape(-1,1,2) 
# cv2.polylines(canvas, [pts], True,(255,0,255),5)

# cv2.imshow('canvas', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

### Circles ### 
canvas = 255*np.ones((600,900,3),dtype='uint8')

cv2.circle(canvas,(100,200),100,(255,0,0),5)
cv2.circle(canvas,(400,200),100,(0,255,0),5)
cv2.circle(canvas,(700,200),100,(0,0,255),5)
cv2.circle(canvas,(250,300),100,(255,0,255),5)
cv2.circle(canvas,(550,300),100,(255,255,0),5)

### text ### 
cv2.putText(canvas, 'Olympics', (200,500), cv2.FONT_HERSHEY_PLAIN,5,(0,0,0),5)

cv2.imshow('canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()