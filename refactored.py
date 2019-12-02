import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


class door_detector:

	def __init__(self,img):
		self.original_img = img

		self.resized = cv2.resize(img, (480, 640)) #resizing to 640 x 480
		self.gray = cv2.cvtColor(self.resized, cv2.COLOR_BGR2GRAY) #converting to grayscale
		self.height, self.width = self.gray.shape
		print "(height, width) (" + str(self.height) + " " + str(self.width) + ")"

		self.denoised = cv2.GaussianBlur(self.gray, (5,5), 0.8) #gaussian blur to remove abrupt corners or noise
		self.thresh = cv2.adaptiveThreshold(self.denoised,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,25) #thresh to change image to binary
		self.edges = cv2.Canny(self.thresh,250,500) #canny edge detection
		self.inv = cv2.bitwise_not(self.thresh)#inverse of the threshold resultant image, black background to subtract the mask.

		#self.current_img = self.denoised	


	def get_image(self):
		return self.denoised


	def feature_extraction(self):
		#converting to np and detecting features
		self.corner_loc = []
		self.edges = np.float32(self.edges)

		self.features = cv2.goodFeaturesToTrack(self.denoised, 500, 0.04, 10)#extract 500 features from the denoised image with 0.04 quality level and 10 as minimum distance
		self.features = np.int0(self.features)

		for ftr in self.features:
			x, y = ftr.ravel() #get x and y from the features

			self.corner_loc.append([x, y])
	
	def choose_features(self):
		#choose suitable features in the image that can compose horizontal lines

		self.points = [] #points taken from features
		self.choosen = [] #already used points

		for point in self.features:
			x1, y1 = point.ravel()  #getting first point

			if x1 < self.width/2: #first point on the first half of the image
				#print x1, y1

				for point2 in self.features:
					x2, y2 = point2.ravel() #getting second point

					if x2 > self.width/2 and abs(y1 - y2) < 10 : #second point on the other half of the image and the difference in height cannot exceed 10 pixels (on the first time it was 5)
						
						#if [x1, y1] not in self.choosen and [x2, y2] not in self.choosen:
						#if [x2, y2] not in self.choosen: #TODO really necessary?

						size = (math.sqrt( (x1 - x2)**2 + (y1 - y2)**2 )) / 800 #size related to the diagonal of an 640 x 480 image

						direc = (math.atan( abs(y1 - y2) / abs(x1 - x2) )) * (180 / math.pi) #direction between the 2 choosen points

						#print size, direc
						#print length
						if size > 0.1 and size < 0.8 and direc < 30:
							

							self.points.append( ((x1,y1),(x2,y2,)) )  #add both points to the list
							
							self.choosen.append([x1,y1])
							self.choosen.append([x2,y2])
	
		self.points = sorted(self.points, key = lambda x: x[0][0]) #sort it from its x coord
		#print self.points
		print "Number of horizontal lines: " + str(len(self.points))
		return self.points


	def choose_squares(self):
		#function that connects horizontal lines with vertical lines to form possible squares

		choosen = []
		#print self.points
		self.square_coords =[]

		for line in self.points:
			#first line (2 points)
			x1 = line[0][0]
			y1 = line[0][1]
			x2 = line[1][0]
			y2 = line[1][1]

			for next_line in self.points:
				#next line
				x3 = next_line[0][0]
				y3 = next_line[0][1]
				x4 = next_line[1][0]
				y4 = next_line[1][1]

				if ([x1, y1] != [x3, y3] and [x2, y2] != [x4, y4]): #do not take the same line

					if(abs(x1 - x3) < 20): #and abs(y1-y3) > self.height * 0.7: #in the first time the parameter was 10

						#getting size and direction of vertical lines 
						size1 = (math.sqrt( (x1 - x3)**2 + (y1 - y3)**2 )) / 800
						size2 = (math.sqrt( (x2 - x4)**2 + (y2 - y4)**2 )) / 800

						direc1 = self.calculate_direc(x1, y1, x3,y3)
						direc2 = self.calculate_direc(x2, y2, x4, y4)

						#direction 0 is also 90 degrees in this case once both lines can have the same y point
						if direc1 == 0:
							direc1 = 90
						if direc2 == 0:
							direc2 = 90
						#print "oi"
						#print line, next_line, size1, size2, direc1, direc2
						if size1 > 0.5 and size1 < 0.9 and size2 > 0.5 and size2 < 0.9:

							if (direc1 > 80 ) and (direc2 > 80):#
								#print "oi"
								if (abs(direc1 - direc2) < 6):
									#print "oi"
									#calculating ratio between the line of the door,vertical lines should be bigger than horizontal
									size3 = (math.sqrt( (x1 - x2)**2 + (y1 - y2)**2 )) / 800
									size4 = (math.sqrt( (x3 - x4)**2 + (y3 - y4)**2 )) / 800	
									ratio = (size1 + size2) / (size3 + size4)
									#print ratio
									if ratio > 1 and ratio < 3:
										
										self.square_coords.append(((x1,y2),(x2,y2),(x3,y3), (x4,y4))) #saving 4 points of the square in a list
										#print line, next_line

				#print (line, next_line)
		print "Number of rectangles: " + str(len(self.square_coords))

	def calculate_direc(self, x1, y1, x2, y2):
		y = y1 - y2
		x = x1 - x2
		if (x1 - x2) == 0:
			x = 1

		return (math.atan( abs(y) / abs(x) )) * (180 / math.pi)


	def draw(self):
		#function that selects the best matching square and draws it
		most_ones = np.zeros((1, 1, 1), dtype = "uint8") #variable to save the best maching square
		mask = None

		for square in self.square_coords:
			#img_door = self.current_img
			img_door = np.zeros((640, 480, 1), dtype = "uint8")  #black mask with same image size

			x1 = square[0][0]
			y1 = square[0][1]
			x2 = square[1][0]
			y2 = square[1][1]
			x4 = square[2][0]
			y4 = square[2][1]
			x3 = square[3][0]
			y3 = square[3][1]

			#draw corners
			cv2.circle(img_door, (x1, y1), 1, 255, 3)
			cv2.circle(img_door, (x2, y2), 1, 255, 3)
			cv2.circle(img_door, (x3, y3), 1, 255, 3)
			cv2.circle(img_door, (x4, y4), 1, 255, 3)

			#draw lines
			cv2.line(img_door, (x1, y1), (x2, y2), (255, 255, 255), 3)
			cv2.line(img_door, (x2, y2), (x3, y3), (255, 255, 255), 3)
			cv2.line(img_door, (x3, y3), (x4, y4), (255, 255, 255), 3)
			cv2.line(img_door, (x4, y4), (x1, y1), (255, 255, 255), 3)

			#draw rectangle
			pts = np.array([[x1,y1],[x2,y2], [x3, y3], [x4,y4] ] )

			#and function between inverted threshold image and the rectangle drawn
 			#the best machting rectangle over the door frame should return the highest number of 1s
			img2 = cv2.bitwise_and(self.inv, img_door)

			#checking number of 1s on the resultant image and picking the one that has the most
			if img2.sum() > most_ones.sum():
				#print img2.sum()
				most_ones = img2
				mask = img_door
			#cv2.imshow('corner door', img2)

			#cv2.waitKey()
		return mask

img = cv2.imread('door_image16.jpg')

detector = door_detector(img)

detector.feature_extraction()
pontos = detector.choose_features()

detector.choose_squares()
#print pontos

img2 = detector.get_image()

mask = detector.draw()

#cv2.imshow("img", detector.inv)
cv2.imshow('corner door', cv2.add(detector.denoised, mask))
cv2.waitKey()