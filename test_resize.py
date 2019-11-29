import cv2

import numpy as np

import math

original_img =cv2.imread(r"door.jpg")

resized = cv2.resize(original_img, (640, 480))

grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
gaussian = cv2.GaussianBlur(grey, (5,5), 0.8)

fast = cv2.FastFeatureDetector_create()

kp = fast.detect(gaussian,None)

# # find and draw the keypoints
# kp = fast.detect(gaussian,None)

reduced_points = []
for i in range (0, len(kp) - 1):
	for j in range (i + 1, len(kp)):
		point1 = kp[i].pt
		point2 = kp[j].pt
		dist = math.sqrt( pow((point1[0] - point2[0]), 2 ) + pow((point1[1] - point2[1]), 2 ))
		if(dist < 10.0):
			if kp[j] not in reduced_points:
				reduced_points.append(kp[j])

d = 800

# # print len(kp)
# kp= []
# for i in range (0,10):
# 	kp.append(i)

# for i in range (0, len(kp) - 3):
# 	for j in range (i + 1, len(kp) - 2):
# 		for k in range (j + 1, len(kp) - 1):
# 			for l in range (k + 1, len(kp)):
# 				# point1 = kp[i].pt
# 				# point2 = kp[j].pt
# 				# point3 = kp[k].pt
# 				# point4 = kp[l].pt

# 				print kp[l]
# # 				# siz12 = math.sqrt( pow((point1[0] - point2[0]), 2 ) + pow((point1[1] - point2[1]), 2 ))
# 				# siz12 = siz12 / d

# 				# print siz12 

for i in reduced_points:
	kp.remove(i)

img2 = cv2.drawKeypoints(gaussian, kp, outImage= None, color=(255,0,0))

cv2.imshow("test", img2)

cv2.waitKey()