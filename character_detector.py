import cv2
import numpy as np

def detect(image):
	"""
	performs detection of characters from image
	:param image: numpy.array
	:return coordinates: list of tuples
		coordinates of detected elements
	:return cropped image: list of numpy.arrays
		bounding boxes of detected elements
	"""
	# resize image to a fixed dimension
	img = cv2.resize(image, (500,500))

	# convert the image to grayscale
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# apply threshold to differentiate foreground and background easier
	thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY, 11, 10)

	# invert the colors
	inverted_image = cv2.bitwise_not(thresh)

	# perform dilatation to increase symbol thickness
	dilatation = cv2.dilate(inverted_image, np.ones((2,2)), iterations=3)

	# find contours
	contours, hierarchy = cv2.findContours(dilatation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	coordinates, cropped_images = list(), list()
	for cnt in contours:
		if cv2.contourArea(cnt) > 100:
			x, y, w, h = cv2.boundingRect(cnt)
			# save coordinates
			coordinates.append((x,y))
			# normalize and save cropped image
			normalized = (255 - gray_image[y:y + h, x:x + w]) / 255
			cropped_images.append(normalized)

	return coordinates, cropped_images