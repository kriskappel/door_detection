import cv2
import numpy as np

from matplotlib import pyplot as plt

img =cv2.imread(r"door2.jpg")

img = cv2.resize(img, (0, 0), None, .25, .25)


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    #--- convert to grayscale 

gaussian = cv2.GaussianBlur(gray, (5,5), 0.8)

th = cv2.adaptiveThreshold(gaussian,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,5,5)

edges = cv2.Canny(th,250,500)


imcont, cnts, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# for contour in cnts:
# 	print cnts
   #cv2.drawContours(img, contour, -1, (0, 255, 0), 3)

img2 = img.copy()

largest = None
for contour in cnts:
    approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour,True),True)
    #print approx
    if len(approx) >= 20:
        #triangle found
        print "square found"
        largest = contour
        cv2.drawContours(img2, [largest], 0, (0,0,255), 3)
        cv2.imshow('img', img2)
        cv2.waitKey()
    	#if largest is None or cv2.contourArea(contour) > cv2.contourArea(largest):
			#largest = contour



# imcont, cnts2, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# rects = []
# for c in cnts:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#     print approx

# img1 = img.copy()

# img2 = img.copy()

# for contour in cnts:
#    cv2.drawContours(img1, contour, -1, (0, 255, 0), 3)


# for contour in cnts2:
#    cv2.drawContours(img2, contour, -1, (0, 255, 0), 3)


# images1 = np.concatenate((th, edges), axis = 1)

# images2 = np.concatenate((img1, img2), axis = 1)
# plt.imshow(images2),plt.show()

# cv2.imshow('img', th)
# cv2.waitKey()
