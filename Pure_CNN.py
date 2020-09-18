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

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

input_shape = (28, 28, 1)

#Normalization of Intensity function
x_train /= 255
x_test /=255

NAME = "CNN_baseModel"

model = Sequential()

model.add(Conv2D(28, (3, 3), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss = 'sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

model.fit(x_train, y_train, epochs=10, validation_split=0.3, callbacks=[tensorboard])
