from keras.applications.resnet50 import ResNet50
import numpy as np
import keras
input_layer = keras.Input(shape=(1,28,28))
inputs = keras.layers.Conv2D(30, (3, 3), activation='relu', data_format='channels_first')(input_layer)
inputs = keras.layers.MaxPooling2D(pool_size=(2, 2))(inputs)
inputs = keras.layers.Conv2D(15, (3,3), activation='relu')(inputs)
inputs = keras.layers.MaxPooling2D(pool_size=(2, 2))(inputs)
inputs = keras.layers.Dropout(0.2)(inputs)
inputs = keras.layers.Flatten()(inputs)
inputs = keras.layers.Dense(128, activation='relu')(inputs)
inputs = keras.layers.Dense(80, activation='relu')(inputs)
inputs = keras.layers.Dense(80, activation='relu')(inputs)
outputs = keras.layers.Dense(22, activation='softmax')(inputs)
model = keras.models.Model(inputs=input_layer,outputs=outputs)
input = np.zeros((1,1,28,28))
print(model.predict(input).shape)
model.save('simpleconv2d_model.hdf5')