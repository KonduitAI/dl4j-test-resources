import onnx

from onnx_tf.backend import prepare
import tensorflow.keras as keras

inp = keras.layers.Input(shape=(10, 10, 10, 1))  # does not work
x = keras.layers.Flatten()(inp)
out = keras.layers.Dense(10)(x)
model = keras.models.Model(inputs=inp,outputs=out)
keras.models.save_model(model,'flatten.hdf5')