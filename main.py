import base64
import io
import cv2
from character_detector import detect
from character_classifier import train_classifier, load_dataset, get_label_for_integer
from math_solver import make_equation, solve
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, Response
from PIL import Image

classifier = tf.keras.models.load_model('model')
app = Flask(__name__, instance_relative_config=True)


@app.route('/')
def start():
	"""
	Initial mapping just for displaying page
	:return: render basic template
	"""
	return render_template("base.html")


@app.route('/script', methods=['POST'])
def solve_equation():
	"""
	Mapping for receiving image and sending back result
	:return: dict
		e : classified equation from image
		r : result of solving equation
	"""
	# get string base64 image
	encoded_data = request.form['img']

	# decode it to bytes
	img_data = base64.b64decode(encoded_data)

	# get opencv image from bytes
	image = Image.open(io.BytesIO(img_data))
	img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

	# get coordinates and cropped images
	coordinates, cropped_images = detect(img)

	# classify all characters in cropped images
	characters = list()
	for img in cropped_images:
		int_prediction = np.argmax(classifier.predict(np.expand_dims(img, axis=(0, 3))))
		characters.append(get_label_for_integer(int_prediction))

	# get equation
	equation = make_equation(coordinates, characters)

	# solve it
	try:
		result = solve(equation)
	except Exception as e:
		# in case of wrongly defined equation
		return {'e': equation, 'r': str(e)}


	# send back result and classified equation
	return {'e':equation, 'r':result}


if __name__ == '__main__':
	app.run(debug=False)