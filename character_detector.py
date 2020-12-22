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
	# convert the image to grayscale
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# apply threshold to differentiate foreground and background easier
	thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY, 21, 9)

	# invert the colors
	inverted_image = cv2.bitwise_not(thresh)

	# perform dilatation to increase symbol thickness
	dilatation = cv2.dilate(inverted_image, np.ones(shape=(2,2)), iterations=5)

	# find contours
	contours, hierarchy = cv2.findContours(dilatation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	coordinates, cropped_images = list(), list()
	image_w, image_h = image.shape[:2]  # image width and height
	for cnt in contours:
		if cv2.contourArea(cnt) >= image_w*image_h*0.0005:  # contours with area more than 0.05% of picture
			x, y, w, h = cv2.boundingRect(cnt)
			# save coordinates
			coordinates.append((x,y))

			# normalize cropped image
			normalized = (255 - gray_image[y:y + h, x:x + w]) / 255

			# make pixels more extreme
			normalized[normalized < 0.4] = 0
			normalized[normalized >= 0.7] = 1

			# resize to 30x30 because that's the model input but keep aspect ratio
			# if image has bigger width then apply padding up and down, if it has bigger height apply it right and left
			if w>h:
				# apply padding with constant color of minimal pixel value
				padded = cv2.copyMakeBorder(normalized, int((w - h) / 2), int((w - h) / 2), 0, 0,
				                            borderType=cv2.BORDER_CONSTANT, value=np.min(normalized))
			else:
				padded = cv2.copyMakeBorder(normalized, 0, 0, int((h - w) / 2), int((h - w) / 2),
				                            borderType=cv2.BORDER_CONSTANT, value=np.min(normalized))

			# erode picture because dataset images are much thinner
			eroded = cv2.morphologyEx(padded, cv2.MORPH_ERODE, np.ones(shape=(2, 2)), iterations=3)

			# resize
			resized = cv2.resize(eroded, (30, 30), interpolation=cv2.INTER_AREA)

			# save cropped images
			cropped_images.append(resized)

	return coordinates, cropped_images