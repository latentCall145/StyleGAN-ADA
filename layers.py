from tensorflow.keras.layers import *
import keras.backend as K
import tensorflow as tf
import numpy as np
import os, time, gc, random

# blocks used in GAN

def minibatchStd(inputs):
    inputs = tf.transpose(inputs, (0, 3, 1, 2)) # NHWC -> NCHW
    group_size = tf.minimum(4, tf.shape(inputs)[0])             # Minibatch must be divisible by (or smaller than) group_size.
    s = inputs.shape                                             # [NCHW]  Input shape.
    y = tf.reshape(inputs, [group_size, -1, 1, s[1], s[2], s[3]])   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  Calc variance over group.
    y = tf.sqrt(y + eps)                                    # [MncHW]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)      # [Mn111]  Take average over fmaps and pixels.
    y = tf.reduce_mean(y, axis=[2])                         # [Mn11] Split channels into c channel groups
    y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [NnHW]  Replicate over group and pixels.
    y = tf.concat([inputs, y], axis=1)                        # [NCHW]  Append as new fmap.
    y = tf.transpose(y, (0, 2, 3, 1)) # NCHW -> NHWC
    return y

class DiffUS(tf.keras.layers.Layer):
    def __init__(self):
        return super().__init__()
    
    def call(self, inputs):
        _N, H, W, C = inputs.shape.as_list()
        x = K.reshape(inputs, (-1, H, 1, W, 1, C))
        x = tf.tile(x, (1, 1, 2, 1, 2, 1))
        used = K.reshape(x, (-1, H * 2, W * 2, C))
        return used

def crop_to_fit(x):
    noise, img = x
    height = img.shape[1]
    width = img.shape[2]
    
    return noise[:, :height, :width, :]

ndist = tf.random_normal_initializer(0, 1)
zeros = tf.zeros_initializer()
ones = tf.ones_initializer()

class FCE(Dense): # fully connected equalized
    def __init__(self, units, kernel_initializer=ndist, bias_initializer=zeros, lrelu=True, *args, **kwargs):
        super().__init__(units, *args, **kwargs)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.lrelu = lrelu
        self.scale = 1

    def build(self, input_shape):
        super().build(input_shape)
        #print('fce', input_shape)
        n = input_shape[-1] # input_shape = (None, features_in) or (None, dimY, dimX, features_in)
        if self.lrelu:
            self.scale = np.sqrt((1 / 0.6) / n) # he but not really, 1 / 0.6 since lrelu(0.2) makes scales variance to 0.6 (0.2 if neg, 1 if pos, div by 2) and you want them to be 1
        else:
            self.scale = np.sqrt(1 / n)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.scale)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not tf.keras.activations.linear:
            output = self.activation(output)
        elif self.lrelu:
            output = LeakyReLU(alpha=0.2)(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'kInit': self.kernel_initializer,
            'bInit': self.bias_initializer,
            'scale': self.scale,
            'useLReLU': self.lrelu,
                      })
        return config

class CVE(Conv2D):
    def __init__(self, units, kernel_size=3, kernel_initializer=ndist, bias_initializer=zeros, padding='same', lrelu=True, *args, **kwargs):
        super().__init__(units, kernel_size, *args, **kwargs)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.padding = padding
        self.lrelu = lrelu
        self.scale = 1

    def build(self, input_shape):
        super().build(input_shape)
        #print('cve', self.kernel.shape)
        n = np.prod(self.kernel.shape[:-1]) # self.kernel.shape = (kernel_x, kernel_y, features_in, features_out)
        if self.lrelu: # he
            self.scale = np.sqrt((1 / 0.6) / n)
        else:
            self.scale = np.sqrt(1 / n)


    def call(self, inputs):
        output = K.conv2d(inputs, self.kernel * self.scale, padding=self.padding)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not tf.keras.activations.linear:
            output = self.activation(output)
        elif self.lrelu:
            output = LeakyReLU(alpha=0.2)(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'kInit': self.kernel_initializer,
            'bInit': self.bias_initializer,
            'padding': self.padding,
            'scale': self.scale,
            'useLReLU': self.lrelu,
                      })
        return config

class ConvMod(Layer):
    def __init__(self, nf, x, w, kSize=3, demod=True):
        super().__init__()
        self.nf = nf
        self.kSize = kSize
        self.xShape = x.shape
        self.wShape = w.shape
        self.scale = FCE(self.xShape[-1], bias_initializer=ones, lrelu=False)
        self.conv = CVE(nf, kSize, lrelu=demod)
        self.conv(x) # create kernel without doing it in build method so h5py doesn't go sicko mode
        self.demod = demod

    def build(self, input_shape): # input_shape: [TensorShape([None, 4, 4, 256]), TensorShape([None, 256]), TensorShape([None, 4, 4, 1])]
        super().build(input_shape)

    def call(self, inputs):
        x, w = inputs

        x = tf.transpose(x, (0, 3, 1, 2)) # NHWC -> NCHW
        weight = self.conv.kernel[np.newaxis] * self.conv.scale # kkio -> 1kkio (1, kernel_size, kernel_size, input_features, output_features)

        scale = self.scale(w)
        scale = scale[:, np.newaxis, np.newaxis, :, np.newaxis] # Bs -> B, 1, 1, s, 1 (s - scaling factor)

        wp = weight * scale # 1kkio * B11s1 -> Bkk(s*i)o
        wpp = wp

        if self.demod:
            wStd = tf.math.rsqrt(tf.reduce_sum(tf.math.square(wp), axis=[1,2,3]) + 1e-8) # Bkkio -> Bo
            wpp = wp * wStd[:, np.newaxis, np.newaxis, np.newaxis, :] # [BkkIO] Scale output feature maps.

        x = tf.reshape(x, (1, -1, x.shape[2], x.shape[3])) # N, C, H, W -> 1, (N*C), H, W

        # B, k, k, i, o -> k, k, i, B, o -> k, k, i, (B*o)
        wpp = tf.reshape(tf.transpose(wpp, [1, 2, 3, 0, 4]), [wpp.shape[1], wpp.shape[2], wpp.shape[3], -1])

        x = tf.nn.conv2d(x, wpp, padding='SAME', data_format='NCHW', strides=[1, 1, 1, 1]) # grouped conv
        x = tf.reshape(x, (-1, self.nf, x.shape[2], x.shape[3])) # 1, (N*C), H, W -> N, C, H, W
        x = tf.transpose(x, (0, 2, 3, 1)) # NCHW -> NHWC
        x = K.bias_add(x, self.conv.bias)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_filters': self.nf,
            'kernel_size': self.kSize,
            'xShape': self.xShape,
            'wShape': self.wShape,
            'demodulated': self.demod
                      })
        return config
