import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_dataset(folder):
	"""
	Reads pickled data from folder and saves it in a X,y dataset
	Every class must be pickled separately
	:param folder: string
		path to the folder which contains dataset
	:return X: numpy.ndarray
		examples matrix with dimensions Nx30x30
	:return y: numpy.ndarray
		class labels matrix with dimensions Nx1
	"""
	# init data matrices, number of rows is hardcoded to number of data points
	X = np.ndarray(shape=(186134,30,30), dtype=float)
	y = np.ndarray(shape=(186134,1), dtype=int)

	# offset for indexing X and y
	offset = 0
	for file in os.listdir(folder):
		with open(folder + file, "rb") as pickle_file:
			# get list of all examples for a class
			data = pickle.load(pickle_file)
			for el in data:
				# each example is a list containing image(np.array) and label(string)
				X[offset] = 1 - el[0]   # images have white background and we need black
				y[offset] = get_integer_for_label(el[1])
				offset += 1
	return X,y



def train_classifier(X,y):
	"""
	Trains neural net on X,y and saves it
	:param X: np.ndarray
		data examples
	:param y: np.ndarray
		labels for examples
	:return: null
	"""
	# split the data 20% to test and 80% to train
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

	# expand matrices to (N,w,h,1)
	X_train = np.expand_dims(X_train, axis=3)
	X_test = np.expand_dims(X_test, axis=3)

	# init model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(30, 30, 1)))
	model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(16, activation='softmax'))

	model.compile(optimizer='adam',
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])

	# train it
	model.fit(X_train, y_train, epochs=2)

	# calculate loss and accuracy
	val_loss, val_acc = model.evaluate(X_test, y_test)
	print("validate loss: ", val_loss)
	print("validate acc: ", val_acc)

	# save it in directory named 'model'
	model.save('model')


def get_integer_for_label(label):
	"""
	:param label: string
		label in dataset
	:return: int
		int representation of that label
	"""
	labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', 'div', 'times', '(', ')']
	return labels.index(label)


def get_label_for_integer(integer):
	"""
	:param integer: int
		int that represents label in dataset
	:return: string
		label
	"""
	labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '/', '*', '(', ')']
	return labels[integer]


def train():
	"""
	Helper function for training model
	Loads dataset, trains classifier and saves it in 'model' directory
	"""
	X, y = load_dataset('dataset/')
	train_classifier(X, y)


# to train new model run this script
if __name__ == '__main__':
	train()