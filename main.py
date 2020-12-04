import cv2
from character_detector import detect

def main():
	image = cv2.imread('brojevi_1.png')
	coordinates, cropped_images = detect(image)

if __name__ == '__main__':
	main()