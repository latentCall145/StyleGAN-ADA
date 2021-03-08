import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from arch import loadGen, zdim, imgSize, nBlocks
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import numpy.random as npr 
import tensorflow as tf
import numpy as np
import os, cv2, scipy

mapper, gen = loadGen()

def makeWs(numImgs, retZs=False):
    z1, z2 = npr.randn(2, numImgs, zdim)

    w1 = mapper.predict(z1)
    w2 = mapper.predict(z2)
    ws = [w1 for _ in range(3)] + [w2 for _ in range(4)]
    zs = [z1 for _ in range(3)] + [z2 for _ in range(4)]
    if retZs:
        return ws, zs
    return ws

def truncNorm(shape=(), psi=0.7):
    return truncnorm(-psi, psi).rvs(shape)

def testW():
    imgs = 3
    fig, axes = plt.subplots(nrows=imgs, ncols=2)
    noise = npr.randn(imgs, imgSize, imgSize, 1)
    ws, zs = makeWs(imgs, retZs=True)
    predsZ = gen.predict([*zs, noise])
    predsW = gen.predict([*ws, noise])

    for i in range(imgs):
        axes[i][0].imshow(predsZ[i]/2+0.5)
        axes[i][1].imshow(predsW[i]/2+0.5)
    plt.show()

def testStyles():
    imgs = 3
    fig, axes = plt.subplots(nrows=imgs, ncols=imgs)
    z1, z2 = npr.randn(2, imgs, zdim)
    noise = npr.randn(imgs, imgSize, imgSize, 1)

    w1 = mapper.predict(z1)
    w2 = mapper.predict(z2)

    for i in range(imgs):
        ws = [w1 for _ in range(3)] + [np.tile(w2[i], (3, 1)) for _ in range(4)]
        preds = gen.predict([*ws, noise])
        for j in range(imgs):
            axes[i][j].imshow(preds[j]/2+0.5)
    plt.show()

def testAllStyles():
    imgs = 3
    fig, axes = plt.subplots(nrows=imgs+1, ncols=nBlocks+1)
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.0, hspace=0.0)
    axes[0][0].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    noise = npr.randn(imgs, imgSize, imgSize, 1)
    zStarts = npr.randn(imgs, zdim)
    wStarts = mapper.predict(zStarts)
    wStartsCopy = [wStarts for _ in range(nBlocks)]
    sourceImgs = gen.predict([*wStartsCopy, noise])
    for i in range(imgs):
        axes[i+1][0].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        axes[i+1][0].imshow(sourceImgs[i]/2 + 0.5)

    noise = npr.randn(nBlocks, imgSize, imgSize, 1)
    zStarts2 = npr.randn(nBlocks, zdim)
    wStarts2 = mapper.predict(zStarts2)
    wStarts2Copy = [wStarts2 for _ in range(nBlocks)]
    sourceImgs = gen.predict([*wStarts2Copy, noise])
    for i in range(nBlocks):
        axes[0][i+1].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        axes[0][i+1].imshow(sourceImgs[i]/2 + 0.5)

    noise = npr.randn(imgs, imgSize, imgSize, 1)

    for i in range(nBlocks):
        if i == 0:
            ws = [wStarts] + [np.tile(wStarts2[i], (3, 1)) for _ in range(nBlocks-1)]
        else:
            ws = [np.tile(wStarts2[i], (3, 1)) for _ in range(i-1)] + [wStarts] + [np.tile(wStarts2[i], (3, 1)) for _ in range(nBlocks-i)]

        preds = gen.predict([*ws, noise])
        for j in range(imgs):
            axes[j+1][i+1].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            axes[j+1][i+1].imshow(preds[j]/2 + 0.5)

    plt.show()

def testNoise():
    imgs = 3
    fig, axes = plt.subplots(nrows=imgs, ncols=imgs)
    ws = makeWs(1)
    for i in range(imgs):
        for j in range(imgs):
            noise = npr.randn(1, imgSize, imgSize, 1)
            preds = gen.predict([*ws, noise])
            axes[i][j].imshow(preds[0]/2+0.5)
    plt.show()

def testTNorm():
    imgs = 3
    fig, axes = plt.subplots(nrows=imgs, ncols=imgs)

    for i in range(imgs):
        z1, z2 = truncNorm((2, imgs, zdim))
        noise = np.random.randn(imgs, imgSize, imgSize, 1)

        w1 = mapper.predict(z1)
        w2 = mapper.predict(z2)
        ws = [w1 for _ in range(3)] + [w2 for _ in range(4)]
        preds = gen.predict([*ws, noise])
        for j in range(imgs):
            axes[i][j].imshow(preds[j]/2+0.5)
    plt.show()

def testMapper():
    z = npr.randn(1, zdim)
    w = mapper.predict(z)
    fig, axes = plt.subplots(nrows=1, ncols=2)

    zDisp = axes[0].imshow(z[0].reshape(16, 16))
    fig.colorbar(zDisp, ax=axes[0])
    wDisp = axes[1].imshow(w[0].reshape(16, 16))
    fig.colorbar(wDisp, ax=axes[1])
    plt.show()

testW()
testStyles()
testAllStyles()
testNoise()
testTNorm()
testMapper()
