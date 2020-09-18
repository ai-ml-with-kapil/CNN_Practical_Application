import tensorflow as tf
tf.__version__

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

input_shape = (28*28,)

#Normalization of Intensity function
x_train /= 255
x_test /=255

NAME = "FlattenLayer_BaseModel"

model = Sequential()
model.add(Dense(28*28, input_shape = input_shape, activation='relu'))
model.add(Dense(10, activation='softmax'))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss = 'sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_split=0.3, callbacks=[tensorboard])
