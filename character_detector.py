import cv2

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

	# find contours
	contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	coordinates, cropped_images = list(), list()
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		# save coordinates
		coordinates.append((x,y))
		# save cropped image
		cropped_images.append(image[y:y + h, x:x + w])

	return coordinates, cropped_images