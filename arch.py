from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from layers import *
import keras.backend as K
import matplotlib.pyplot as plt
import numpy.random as npr
import tensorflow as tf
import numpy as np
import os, time, gc, random

zdim = 256
imgSize = 256
nBlocks = int(np.log2(imgSize // 4)) + 1

def gblock(accum, x, w, noiseInp, filters, us=True):
    if us:
        x = DiffUS()(x)
        accum = DiffUS()(accum)
    
    for i in range(2):
        x = ConvMod(filters, x, w)([x, w])
        noise = Lambda(crop_to_fit)([noiseInp, x])
        noise = FCE(filters, kernel_initializer=zeros, use_bias=False, lrelu=False)(noise)
        x = Add()([x, noise])
        x = LeakyReLU(alpha=0.2)(x)
    
    trgb = ConvMod(3, x, w, 1, demod=False)([x, w])
    accum = Add()([accum, trgb]) * np.sqrt(1 / 2)
        
    return accum, x

def dblock(x, filters, maxFilters=256):
    frgb = CVE(min(2 * filters, maxFilters), 1, lrelu=False, use_bias=False)(x)
    
    x = CVE(filters)(x)
    x = CVE(min(2 * filters, maxFilters))(x)
        
    frgb = AveragePooling2D()(frgb)
    x = AveragePooling2D()(x)
    x = Add()([x, frgb]) * (1 / np.sqrt(2))
    
    return x

def ztow(nlayers=8):
    z = Input((zdim,))
    w = z
    if nlayers > 0:
        w = LayerNormalization()(w)
    for i in range(max(nlayers-1, 0)):
        w = FCE(zdim)(w)
    return Model(z, w, name='mapping')

def genGen():
    ws = [Input((zdim,), name='w{}'.format(i)) for i in range(nBlocks)]
    noiseInp = Input((imgSize, imgSize, 1), name='noiseInp')

    x = Dense(1)(ws[0]); x = Lambda(lambda x: x * 0 + 1)(x)
    x = FCE(4*4*zdim, lrelu=False, use_bias=False)(x)
    x = Reshape((4, 4, zdim))(x)

    #layerFilters = (16, 16, 16, 16, 8, 4)
    layerFilters = (256, 256, 256, 128, 64, 32)

    x = ConvMod(layerFilters[0], x, ws[0])([x, ws[0]])
    noise = Lambda(crop_to_fit)([noiseInp, x])
    noise = FCE(layerFilters[0], kernel_initializer=zeros, use_bias=False, lrelu=False)(noise)
    x = Add()([x, noise])
    x = LeakyReLU(alpha=0.2)(x)

    accum = ConvMod(3, x, ws[0], 1, demod=False)([x, ws[0]])
    for idx, f in enumerate(layerFilters):
        accum, x = gblock(accum, x, ws[idx+1], noiseInp, f)

    # Final CNN layer
    out = CVE(3, 1, lrelu=False)(accum)
    return Model([*ws, noiseInp], out, name='generator')

def genDisc():
    inp = Input((imgSize, imgSize, 3)); x = inp

    layerFilters = (32, 64, 128, 256, 256, 256)
    #layerFilters = (4, 8, 16, 16, 16, 16)

    x = CVE(layerFilters[0], 1)(x)
    for fi, f in enumerate(layerFilters):
        x = dblock(x, f, maxFilters=layerFilters[-1])

    x = Lambda(minibatchStd)(x)
    x = CVE(layerFilters[-1])(x)
    x = Flatten()(x)
    x = FCE(layerFilters[-1])(x)
    out = FCE(1, lrelu=False)(x)

    return Model(inp, out, name='discriminator')

def loadGen():
    gen = genGen()
    mapper = ztow()
    gen.load_weights('models/genWeights.h5')
    mapper.load_weights('models/mapWeights.h5')
    return mapper, gen
