import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import RMSprop


class Test(unittest.TestCase):
	def setUp(self):
		import Exercise10_02
		self.exercises = Exercise10_02

		np.random.seed(8)
		tf.random.set_seed(8)

		self.model = tf.keras.Sequential()

		conv1 = Conv2D(32, 8, (4, 4), activation='relu', input_shape=(84, 84, 1))
		conv2 = Conv2D(64, 4, (2, 2), activation='relu')
		conv3 = Conv2D(64, 3, (1, 1), activation='relu')
		fc1 = Dense(256, activation='relu')
		fc2 = Dense(4)

		self.model.add(conv1)
		self.model.add(conv2)
		self.model.add(conv3)
		self.model.add(fc1)
		self.model.add(fc2)

		optimizer = RMSprop(lr=0.00025)

		self.model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

	def test_summary(self):
		self.assertEqual(self.exercises.model.summary(), self.model.summary())


if __name__ == '__main__':
	unittest.main()