import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from arch import loadGen, zdim, imgSize, nBlocks
import matplotlib.pyplot as plt
import numpy.random as npr 
import tensorflow as tf
from PIL import Image
import numpy as np

def pad(number, strLen=5):
    padded = str(number)
    padStart = len(padded)
    for k in range(strLen - padStart):
        padded = '0' + padded 

    return padded

numFrames = 30
digLen = len(str(numFrames))

try:
    os.mkdir('noiseImgs')
except:
    pass

mapper, gen = loadGen()
z = npr.randn(1, zdim)
w = mapper.predict(z)
ws = [w for _ in range(nBlocks)]

for i in range(numFrames):
    noise = npr.randn(1, imgSize, imgSize, 1)
    pred = gen.predict([*ws, noise])[0]/2 + 0.5
    pred[pred < 0] = 0
    pred[pred > 1] = 1
    img = Image.fromarray((255 * (pred)).astype(np.uint8))
    padded = pad(i, digLen)
    img.save('noiseImgs/{}.png'.format(padded))
