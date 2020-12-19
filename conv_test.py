import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense

'''
 * @author Nassim MOUALEK
 * @since 06/12/2020
'''

if __name__ == '__main__':
    def create_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(24, kernel_size=(3, 3), strides=(3, 3), activation='relu', name='conv_1', input_shape=(20, 20, 3)),
            tf.keras.layers.Dense(4, activation='relu')
        ])
        return model


    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
