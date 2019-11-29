import cv2
import numpy as np

img =cv2.imread(r"door2.jpg")

img = cv2.resize(img, (0, 0), None, .25, .25)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    #--- convert to grayscale 

gaussian = cv2.GaussianBlur(gray, (5,5), 0.8)

th = cv2.adaptiveThreshold(gaussian,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,5,5)

edges = cv2.Canny(th,250,500)

corners = cv2.goodFeaturesToTrack(edges, 10, 0.1, 100)
corners = np.int0(corners)

img2=img.copy()

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),3,255,-1)
    
cv2.imshow('Corner',img)
cv2.waitKey(0)