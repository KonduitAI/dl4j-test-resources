from keras.applications.resnet50 import ResNet50
import numpy as np
import keras
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(30, (3, 3), activation='relu', data_format='channels_first', input_shape=(1, 28, 28)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(15, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(80, activation='relu'))
model.add(keras.layers.Dense(80, activation='relu'))
model.add(keras.layers.Dense(22, activation='softmax'))
model.save('simpleconv2d.hdf5')