import cv2
import numpy as np
from matplotlib import pyplot as plt

original_img =cv2.imread(r"door.jpg")

resized = cv2.resize(original_img, (0, 0), None, .25, .25)

grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

img = grey

# blur = cv2.blur(grey, (3,3))

gaussian = cv2.GaussianBlur(grey, (5,5), 0.8)

# median = cv2.medianBlur(grey, 5)

edges = cv2.Canny(gaussian,50,150)

edges1 = cv2.Canny(grey,50,150, apertureSize = 3)


# kernel = np.ones((2,2),np.uint8)
# abertura = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel)

# thresh = cv2.threshold(gaussian, 100, 255, cv2.THRESH_BINARY)[1]
# #grey = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2BGR)

#grayscale_img =cv2.fastNlMeansDenoisingColored(my,None,5,5,7,21)


images1 = np.concatenate((edges, edges1), axis = 1)

# images2 = np.concatenate((blur, edges), axis = 1)

# combined = np.concatenate((images1, images2), axis = 0)


# corners = cv2.goodFeaturesToTrack(gaussian,25,0.01,10)
# corners = np.int0(corners)

# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(resized,(x,y),3,255,-1)

# plt.imshow(resized),plt.show()

# lines = cv2.HoughLines(edges,1,np.pi/180,0)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))

#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imwrite('houghlines3.jpg',img)
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(gaussian,None)


# print kp

# for i in kp:
# 	print i.pt
# img2 = cv2.drawKeypoints(gaussian, kp, outImage= None, color=(255,0,0))


# cv2.imshow("test", img2)

# cv2.waitKey()

test = []

for i in range (0,10):
	test.append(i)

print test

for i in range (0, len(test) - 3):
	for j in range (i + 1, len(test) - 2):
		for k in range (j + 1, len(test) - 1):
			for l in range (k + 1, len(test)):
				print (test[i], test[j], test[k], test[l])