
```
from __future__ import print_function
import numpy as np
import keras
from keras.layers import Conv2DTranspose, Input
from keras.models import Model
from random import seed
from random import random
from random import randint

kernels = [1, 2, 3, 4]
inSizes = [3, 8]
strides = [1, 2, 3]
pad = ["valid", "same"]
nchw = [True, False]
dilation = [1, 2]
mb = [1, 3]

chIn = 2
chOut = 3


seed(12345)
rootdir = "C:/Temp/deconv/"


count = 0;
for k in kernels:
    for i in inSizes:
        for s in strides:
                for p in pad:
                    for d in dilation:
                        for f in nchw:
                            for m in mb:
                                if (d != 1 and s != 1): #"Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1."
                                    continue
                                if randint(0,10) is not 0:
                                    continue
                                name = "mb" + str(m) + "_k" + str(k) + "_sz" + str(i) + "_s" + str(s) + "_" + p + "_d" + str(d) + ("_nchw" if f else "_nhwc")
                                print(name)
                                count += 1
                                if f is True:
                                    input_shape = (chIn, i, i)
                                    df = "channels_first"
                                    inArr = np.arange(chIn*i*i).reshape(1,chIn,i,i)
                                else:
                                    input_shape = (i, i, chIn)
                                    df = "channels_last"
                                    inArr = np.arange(chIn*i*i).reshape(1,i,i,chIn)
                                if p is "valid":
                                    output_padding = (0,0)
                                else:
                                    output_padding = None
                                inputs = Input(shape=input_shape, name='in')
                                outputs = Conv2DTranspose(filters=chOut,kernel_size=(k,k),strides=s,padding=p, data_format=df, \
                                                          dilation_rate=(d, d), kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None), \
                                                          bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),
                                                          output_padding=output_padding)(inputs)
                                model = Model(inputs, outputs, name='out')
                                #print(model.summary())
                                w = model.layers[1].get_weights()[0]
                                b = model.layers[1].get_weights()[1]
                                out = model.predict(inArr)

                                outW = rootdir + name + "_W.npy"
                                outB = rootdir + name + "_b.npy"
                                outIn = rootdir + name + "_in.npy"
                                outOut = rootdir + name + "_out.npy"
                                np.save(outW, w)
                                np.save(outB, b)
                                np.save(outIn, inArr)
                                np.save(outOut, out)


print(count)
```
