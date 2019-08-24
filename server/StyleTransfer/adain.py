# -*- coding: utf-8 -*-
"""

(c) shwu, 2019

(c) ytlee, 2019
"""


import os

#==================================================
#
#  Third-party Libraries
#
#==================================================
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import VGG19

from PIL import Image
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import cv2

#==================================================
#
#  .data
#
#==================================================
_V19In = 0x100 #0xE0
_V19InSq = _V19In * _V19In
_V19InHl = _V19In >> 0x1
_V19Out = _V19In >> 0x3

# VGG19 was trained by Caffe which converted images from RGB to BGR,
# then zero-centered each color channel with respect to the ImageNet 
# dataset, without scaling.  
_V19Mn = np.array([103.939, 116.779, 123.68], dtype = np.float32) # BGR
_V19Lo = -_V19Mn 
_V19Hi = 255.0 - _V19Mn 
_tfV19Mn = tf.convert_to_tensor(_V19Mn)
_tfV19Lo = tf.convert_to_tensor(_V19Lo)
_tfV19Hi = tf.convert_to_tensor(_V19Hi)
_V19HW = (_V19In, _V19In)
_V19HWhlf = (_V19InHl, _V19InHl)
_V19HWC = (_V19In, _V19In, 0x3)
_V19XSC = (-0x1, _V19InSq, 0x3)
_V19NSC = (None, _V19InSq, 0x3)
_V19BHWC = (0x1, _V19In, _V19In, 0x3)
_V19XHWC = (-0x1, _V19In, _V19In, 0x3)
_V19NHWC = (None, _V19In, _V19In, 0x3)
_V19B4C1 = (_V19Out, _V19Out, 0x200)
_V19B4C1N = (None, _V19Out, _V19Out, 0x200)

_YIQMl = np.array([0.114, 0.587, 0.299], dtype = np.float32) # BGR
_YIQBa = np.dot(_YIQMl, _V19Mn)
_YIQAx = (0x3, 0x0)
_tfYIQMl = tf.convert_to_tensor(_YIQMl)
_tfYIQBa = tf.convert_to_tensor(_YIQBa)

_instV19 = VGG19(include_top = False, weights = "imagenet", input_shape = _V19HWC)
_instV19.trainable = False
laV19GetLayer = _instV19.get_layer

_ADA_TKN = "in_ada"

#==================================================
#
#  class
#
#==================================================

class crc32r:
    def __init__(self, p0 = 0xEDB88320):
        p0 |= 0x80000000 # CRC Polynomial 必須有常數項
        u0 = [0x0] * 0x100
        u1 = [0x0] * 0x100
        p1 = 0x1 | ((p0 & 0x7FFFFFFF) << 0x1)
        i = 0x1
        while i & 0xFF:
            t0 = i
            t1 = i << 0x18
            for j in range(0x8):
                b = bool(t0 & 0x1)
                t0 >>= 0x1
                if b : t0 ^= p0
                b = bool(t1 >> 0x1F)
                t1 <<= 0x1
                if b : t1 &= 0xFFFFFFFF; t1 ^= p1
            u0[i] = t0
            u1[i] = t1
            i += 0x1
        self.u0 = tuple(u0)
        self.u1 = tuple(u1)
        u0.clear()
        u1.clear()
            
    step0 = lambda self, src, key : (src >> 0x8) ^ self.u0[(key ^ src) & 0xFF]
    
    step1 = lambda self, src, key : key ^ ((src & 0x00FFFFFF) << 0x8) ^ self.u1[(src >> 0x18)]
    
    def calc0(self, s, inXOR = 0xFFFFFFFF, outXOR = 0xFFFFFFFF):
        u = self.u0
        t = inXOR
        for k in s : t = (t >> 0x8) ^ u[(k ^ t) & 0xFF]
        return t ^ outXOR
    
    def calc1(self, s, inXOR = 0xFFFFFFFF, outXOR = 0xFFFFFFFF):
        u = self.u1
        t = inXOR # calc0 ^ outXOR
        for k in s : t = k ^ ((t & 0x00FFFFFF) << 0x8) ^ u[(t >> 0x18)] # reversed string
        return t ^ outXOR

class laAstBGR(keras.layers.Layer):
    def __init__(
        self,
        clw = 1.0,
        slw = 10.0,
        eps = 1.0e-06,
        name = "ASTBGR",
        **kwargs
    ):
        super(laAstBGR, self).__init__(name = name, **kwargs)
        
        # Hyper-parameters
        self.clw = tf.convert_to_tensor(clw)
        self.slw = tf.convert_to_tensor(slw)
        self.eps = tf.convert_to_tensor(eps)
        
        # Encoder
        _vgg19 = keras.Model(
            inputs = _instV19.input,
            outputs = [
                    laV19GetLayer("block1_conv1").output,
                    laV19GetLayer("block2_conv1").output,
                    laV19GetLayer("block3_conv1").output,
                    laV19GetLayer("block4_conv1").output,
                ],
            name = "vgg19"
        )
        _vgg19.trainable = False
        self.vgg19 = _vgg19
        
        # Decoder
        _fd0 = keras.Input(shape = _V19B4C1, name = _ADA_TKN)
        _d1_1 = keras.layers.Conv2DTranspose(
                    0x100,
                    0x3,
                    padding = "same", 
                    activation = "relu"
        )
        _fd1_1 = _d1_1(_fd0)
        _d1_u = keras.layers.UpSampling2D(0x2)
        _fd1 = _d1_u(_fd1_1)
        _d2_1 = keras.layers.Conv2DTranspose(
                0x100,
                0x3,
                padding = "same", 
                activation = "relu"
        )
        _fd2_1 = _d2_1(_fd1)
        _d2_2 = keras.layers.Conv2DTranspose(
                0x100,
                0x3,
                padding = "same", 
                activation = "relu"
        )
        _fd2_2 = _d2_2(_fd2_1)
        _d2_3 = keras.layers.Conv2DTranspose(
                0x100,
                0x3,
                padding = "same", 
                activation = "relu"
        )
        _fd2_3 = _d2_3(_fd2_2)
        _d2_4 = keras.layers.Conv2DTranspose(
                0x80,
                0x3,
                padding = "same", 
                activation = "relu"
        )
        _fd2_4 = _d2_4(_fd2_3)
        _d2_u = keras.layers.UpSampling2D(0x2)
        _fd2 = _d2_u(_fd2_4)
        _d3_1 = keras.layers.Conv2DTranspose(
                0x80,
                0x3,
                padding = "same", 
                activation = "relu"
        )
        _fd3_1 = _d3_1(_fd2)
        _d3_2 = keras.layers.Conv2DTranspose(
                0x40,
                0x3,
                padding = "same", 
                activation = "relu"
        )
        _fd3_2 = _d3_2(_fd3_1)
        _d3_u = keras.layers.UpSampling2D(0x2)
        _fd3 = _d3_u(_fd3_2)
        _d4_1 = keras.layers.Conv2DTranspose(
                0x40,
                0x3,
                padding = "same", 
                activation = "relu"
        )
        _fd4_1 = _d4_1(_fd3)
        _d4_2 = keras.layers.Conv2DTranspose(
                0x3,
                0x3,
                padding = "same", 
                activation = "relu"
        )
        _fd4_2 = _d4_2(_fd4_1)
        
        decoder = keras.Model(inputs=_fd0, outputs = _fd4_2, name = "decoder")
        decoder._fd0 = _fd0
        decoder._fd1_1 = _fd1_1
        decoder._fd1 = _fd1
        decoder._fd2_1 = _fd2_1
        decoder._fd2_2 = _fd2_2
        decoder._fd2_3 = _fd2_3
        decoder._fd2_4 = _fd2_4
        decoder._fd2 = _fd2
        decoder._fd3_1 = _fd3_1
        decoder._fd3_2 = _fd3_2
        decoder._fd3 = _fd3
        decoder._fd4_1 = _fd4_1
        decoder._fd4_2 = _fd4_2
        decoder._d1_1 = _d1_1
        decoder._d1_u = _d1_u
        decoder._d2_1 = _d2_1
        decoder._d2_2 = _d2_2
        decoder._d2_3 = _d2_3
        decoder._d2_4 = _d2_4
        decoder._d2_u = _d2_u
        decoder._d3_1 = _d3_1
        decoder._d3_2 = _d3_2
        decoder._d3_u = _d3_u
        decoder._d4_1 = _d4_1
        decoder._d4_2 = _d4_2
        _egcall = decoder.__call__
        decoder._egcall = _egcall
        decoder.__call__ = tf.function(
            _egcall,
            input_signature = [
                tf.TensorSpec(_V19B4C1N, tf.float32, name = _ADA_TKN)
            ],
            autograph = False
        )
        self.decoder = decoder
        
    @tf.function(
        input_signature = [
            tf.TensorSpec(_V19NHWC, tf.float32, name = "c"),
            tf.TensorSpec(_V19NHWC, tf.float32, name = "s")
        ],
        autograph = False
    )
    def train(self, c, s):
        v19 = self.vgg19
        
        #Encoder
        ec = v19(c)
        es = v19(s)
        
        # AdaIN
        eps = self.eps
        c0 = ec[0x3]
        s0 = es[0x3]
        uc, vc = tf.nn.moments(c0, axes = (0x1, 0x2), keepdims = True)
        us, vs = tf.nn.moments(s0, axes = (0x1, 0x2), keepdims = True)
        sc = tf.sqrt(vc + eps)
        ss = tf.sqrt(vs + eps)
        nc =  ss * (c0 - uc) / sc + us
        
        # Decoder
        out = self.decoder(nc)
        
        # Clip
        out = tf.clip_by_value(out, _tfV19Lo, _tfV19Hi)
        
        # Loss
        eo = v19(out)
        e0 = eo[0x3]
        
        # Content Loss
        cl = tf.reduce_mean(
            tf.math.squared_difference(e0, nc)
        )
        cl = self.clw * cl
        
        # Style Loss
        uo, vo = tf.nn.moments(e0, axes = (0x1, 0x2), keepdims = True)
        so = tf.sqrt(vo + eps)
        sl = tf.reduce_mean(
          tf.math.squared_difference(uo, us)
        ) + tf.reduce_mean(
          tf.math.squared_difference(so, ss)
        )
        us, vs = tf.nn.moments(es[0x2], axes = (0x1, 0x2), keepdims = True)
        ss = tf.sqrt(vs + eps)
        uo, vo = tf.nn.moments(eo[0x2], axes = (0x1, 0x2), keepdims = True)
        so = tf.sqrt(vo + eps)
        sl += tf.reduce_mean(
          tf.math.squared_difference(uo, us)
        ) + tf.reduce_mean(
          tf.math.squared_difference(so, ss)
        )
        us, vs = tf.nn.moments(es[0x1], axes = (0x1, 0x2), keepdims = True)
        ss = tf.sqrt(vs + eps)
        uo, vo = tf.nn.moments(eo[0x1], axes = (0x1, 0x2), keepdims = True)
        so = tf.sqrt(vo + eps)
        sl += tf.reduce_mean(
          tf.math.squared_difference(uo, us)
        ) + tf.reduce_mean(
          tf.math.squared_difference(so, ss)
        )
        us, vs = tf.nn.moments(es[0x0], axes = (0x1, 0x2), keepdims = True)
        ss = tf.sqrt(vs + eps)
        uo, vo = tf.nn.moments(eo[0x0], axes = (0x1, 0x2), keepdims = True)
        so = tf.sqrt(vo + eps)
        sl += tf.reduce_mean(
          tf.math.squared_difference(uo, us)
        ) + tf.reduce_mean(
          tf.math.squared_difference(so, ss)
        )
        sl = self.slw * sl
        gds = tf.gradients(cl + sl, self.trainable_variables)
        
        return out, gds, cl, sl
    
    @tf.function(
        input_signature = [
            tf.TensorSpec(_V19NHWC, tf.float32, name = "c"),
            tf.TensorSpec(_V19NHWC, tf.float32, name = "s")
        ],
        autograph = False
    )
    def test(self, c, s):
        v19 = self.vgg19
        
        #Encoder
        ec = v19(c)
        es = v19(s)
        
        # AdaIN
        eps = self.eps
        c0 = ec[0x3]
        s0 = es[0x3]
        uc, vc = tf.nn.moments(c0, axes = (0x1, 0x2), keepdims = True)
        us, vs = tf.nn.moments(s0, axes = (0x1, 0x2), keepdims = True)
        sc, ss = tf.sqrt(vc + eps), tf.sqrt(vs + eps)
        nc =  ss * (c0 - uc) / sc + us
        
        # Decoder
        out = self.decoder(nc)
        
        # Clip
        return tf.clip_by_value(out, _tfV19Lo, _tfV19Hi)
    
    def save(self, path):
        self.decoder.save_weights(path)
        
    def load(self, path):
        self.decoder.load_weights(path)
        
        
class MegamiBase():
    def __init__(self, crt, srt, drt = None, cmans = None, smans = None, dmans = None, pair = None):
        h32 = crc32r()
        crc32 = h32.calc1
        cpt = []
        cadd = cpt.append
        if cmans:
            rng = np.random.RandomState(0x0)
            if not os.path.exists(crt) : os.mkdir(crt)
            for cman in cmans:
                cnt = len(cman)
                for rt, ds, fs in os.walk(cman):
                    pfx = rt[cnt :]
                    for fn in fs:
                        fin = rt + fn
                        fout = "%08X" % crc32((pfx + fn).encode("UTF-8"))
                        
                        # Decode Image
                        im0 = cv2.imread(fin)
                        
                        # Test Decode
                        if im0 is None : continue
                        cadd(fout)
                        im0 = im0.astype(np.float32)
                        
                        # Padding
                        s0, s1, s2 = im0.shape
                        if s0 > s1:
                            im1 = np.full((s0, s0, s2), 255.0, dtype = np.float32)
                            a = rng.randint(0x0, s0 - s1)
                            im1[:, a : a + s1, :] = im0[:,:,:]
                        elif s0 < s1:
                            im1 = np.full((s1, s1, s2), 255.0, dtype = np.float32)
                            a = rng.randint(0x0, s1 - s0)
                            im1[a : a + s0, :, :] = im0[:,:,:]
                        else:
                            im1 = im0
                        
                        # Resize
                        im2 = cv2.resize(im1, _V19HW)
                        
                        # To VGG
                        #im = im[..., ::-0x1] # RGB to BGR (No requried with cv2)
                        im2 -= _V19Mn # BGR means
                        open(crt + fout, "wb").write(im2.tobytes())
            #plt.imshow((im2 + _V19Mn)[..., ::-0x1].astype(np.uint8))
        else:
            rt, ds, fs = next(iter(os.walk(crt)))
            for fn in fs : cadd(fn)
        cpt.sort()
        
        spt = []
        sadd = spt.append
        if smans:
            if not os.path.exists(srt) : os.mkdir(srt)
            for sman in smans:
                cnt = len(sman)
                for rt, ds, fs in os.walk(sman):
                    pfx = rt[cnt :]
                    for fn in fs:
                        fin = rt + fn
                        fout = "%08X" % crc32((pfx + fn).encode("UTF-8"))
                        
                        # Decode Image
                        im = cv2.imread(fin)
                        
                        # Test Decode
                        if im is None : continue
                        sadd(fout)
                        im = im.astype(np.float32)
                        
                        # Resize
                        im = cv2.resize(im, _V19HW)
                        #im = im[..., ::-0x1] # RGB to BGR (No requried with cv2)
                        
                        im -= _V19Mn # BGR means
                        open(srt + fout, "wb").write(im.tobytes())
        else:
            rt, ds, fs = next(iter(os.walk(srt)))
            for fn in fs : sadd(fn)
        spt.sort()
        
        ppt = []
        if pair:
            padd = ppt.append
            for fin in cpt:
                for fout in spt : padd(fin + fout)
        
        dpt = []
        if drt:
            dadd = dpt.append
            if dmans:
                if not os.path.exists(drt) : os.mkdir(drt)
                for dman in dmans:
                    cnt = len(dman)
                    for rt, ds, fs in os.walk(dman):
                        pfx = rt[cnt :]
                        for fn in fs:
                            fin = rt + fn
                            fout = "%08X" % crc32((pfx + fn).encode("UTF-8"))
                            im = cv2.imread(fin)
                            if im is None : continue
                            dadd(fout)
                            im = im.astype(np.float32)
                            im = cv2.resize(im, _V19HW)
                            #im = im[..., ::-0x1] # RGB to BGR (No requried with cv2)
                            im -= _V19Mn # BGR means
                            open(drt + fout, "wb").write(im.tobytes())
            else:
                rt, ds, fs = next(iter(os.walk(drt)))
                for fn in fs : dadd(fn)
            dpt.sort()
        
        self.cpt = cpt
        self.nc = len(cpt)
        self.spt = spt
        ns = len(spt)
        self.ns = ns
        self._ns = tf.constant(ns, tf.uint16)
        self._rs = tf.constant(1.0 / float(ns - 0x1), tf.float32)
        self.ppt = ppt
        self.nx = len(ppt)
        self.dpt = dpt
        self.nd = len(dpt)
        self.h32 = h32
        
    def setV19(
            self,
            b1, b2,
            slw,
            epx, eps,
            name
        ):
        mdl = keras.Model(
            inputs = _instV19.input,
            outputs = [
                laV19GetLayer("block1_conv1").output,
                laV19GetLayer("block2_conv1").output,
                laV19GetLayer("block3_conv1").output,
                laV19GetLayer("block4_conv1").output,
                laV19GetLayer("block5_conv1").output,
                laV19GetLayer("block5_pool").output
            ],
            name = name
        )
        self.V19 = mdl
        self._slw = tf.convert_to_tensor(slw)
        _b1 = tf.convert_to_tensor(b1)
        _b2 = tf.convert_to_tensor(b2)
        self._a1 = 1.0 - _b1
        self._a2 = 1.0 - _b2
        self._b1 = _b1
        self._b2 = _b2
        self._epx = tf.convert_to_tensor(epx)
        self._eps = tf.convert_to_tensor(eps)
        
    def setAst(
            self,
            clw, slw,
            epx,
            path,
            name = "ASTBGR"
        ):
        ast = laAstBGR(clw, slw, epx, name)
        
        # Load decoder weight
        if path:
            ckp = tf.train.latest_checkpoint(path)
            if ckp : ast.load(ckp)
            
        self.ast = ast
        
    @tf.function(
        input_signature = [
            tf.TensorSpec(_V19NHWC, tf.float32, name = "c0"),
            tf.TensorSpec(_V19NHWC, tf.float32, name = "c1")
        ],
        autograph = False
    )
    def v19Loss(
            self,
            c0, c1
        ):
            ft0 = self.V19(c0)
            ft1 = self.V19(c1)
            
            eps = self._epx
            
            # Content Correlation Loss
            
            t0 = ft0[0x5]
            u0, v0 = tf.nn.moments(t0, axes = (0x1, 0x2), keepdims = True)
            s0 = tf.sqrt(v0 + eps)
            n0 = (t0 - u0) / s0
            t1 = ft1[0x5]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl = tf.math.abs(tf.reduce_mean(n0 * n1, axis = (0x1, 0x2, 0x3)))
            
            t0 = ft0[0x4]
            u0, v0 = tf.nn.moments(t0, axes = (0x1, 0x2), keepdims = True)
            s0 = tf.sqrt(v0 + eps)
            n0 = (t0 - u0) / s0
            t1 = ft1[0x4]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl += tf.math.abs(tf.reduce_mean(n0 * n1, axis = (0x1, 0x2, 0x3)))
            
            t0 = ft0[0x3]
            u0, v0 = tf.nn.moments(t0, axes = (0x1, 0x2), keepdims = True)
            s0 = tf.sqrt(v0 + eps)
            n0 = (t0 - u0) / s0
            t1 = ft1[0x3]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl += tf.math.abs(tf.reduce_mean(n0 * n1, axis = (0x1, 0x2, 0x3)))
            
            # Style Loss
            
            sl = tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, v0 = tf.nn.moments(ft0[0x2], axes = (0x1, 0x2), keepdims = True)
            s0 = tf.sqrt(v0 + eps)
            u1, v1 = tf.nn.moments(ft1[0x2], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, v0 = tf.nn.moments(ft0[0x1], axes = (0x1, 0x2), keepdims = True)
            s0 = tf.sqrt(v0 + eps)
            u1, v1 = tf.nn.moments(ft1[0x1], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, v0 = tf.nn.moments(ft0[0x0], axes = (0x1, 0x2), keepdims = True)
            s0 = tf.sqrt(v0 + eps)
            u1, v1 = tf.nn.moments(ft1[0x0], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
        
            sl *= self._slw
            
            return fl + sl, fl, sl
        
    @tf.function(
        input_signature = [
            tf.TensorSpec(_V19NHWC, tf.float32, name = "c0"),
            tf.TensorSpec(_V19NHWC, tf.float32, name = "c1")
        ],
        autograph = False
    )
    def v19Grads(
            self,
            c0, c1
        ):
            ft0 = self.V19(c0)
            ft1 = self.V19(c1)
            
            eps = self._epx
            
            # Content Correlation Loss
            
            t0 = ft0[0x5]
            u0, v0 = tf.nn.moments(t0, axes = (0x1, 0x2), keepdims = True)
            s0 = tf.sqrt(v0 + eps)
            n0 = (t0 - u0) / s0
            t1 = ft1[0x5]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl = tf.math.abs(tf.reduce_mean(n0 * n1, axis = (0x1, 0x2, 0x3)))
            
            t0 = ft0[0x4]
            u0, v0 = tf.nn.moments(t0, axes = (0x1, 0x2), keepdims = True)
            s0 = tf.sqrt(v0 + eps)
            n0 = (t0 - u0) / s0
            t1 = ft1[0x4]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl += tf.math.abs(tf.reduce_mean(n0 * n1, axis = (0x1, 0x2, 0x3)))
            
            t0 = ft0[0x3]
            u0, v0 = tf.nn.moments(t0, axes = (0x1, 0x2), keepdims = True)
            s0 = tf.sqrt(v0 + eps)
            n0 = (t0 - u0) / s0
            t1 = ft1[0x3]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl += tf.math.abs(tf.reduce_mean(n0 * n1, axis = (0x1, 0x2, 0x3)))
            
            # Style Loss
            
            sl = tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, v0 = tf.nn.moments(ft0[0x2], axes = (0x1, 0x2), keepdims = True)
            s0 = tf.sqrt(v0 + eps)
            u1, v1 = tf.nn.moments(ft1[0x2], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, v0 = tf.nn.moments(ft0[0x1], axes = (0x1, 0x2), keepdims = True)
            s0 = tf.sqrt(v0 + eps)
            u1, v1 = tf.nn.moments(ft1[0x1], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, v0 = tf.nn.moments(ft0[0x0], axes = (0x1, 0x2), keepdims = True)
            s0 = tf.sqrt(v0 + eps)
            u1, v1 = tf.nn.moments(ft1[0x0], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
        
            sl *= self._slw
            
            return tf.gradients(fl + sl, c1), fl
    
    @tf.function(
        input_signature = [
            tf.TensorSpec(_V19NHWC, tf.float32, name = "c0")
        ],
        autograph = False
    )
    def v19GradsX0(
            self, 
            c0
        ):
            epx = self._epx
            ft = self.V19(c0)
            
            t5 = ft[0x5]
            u5, v5 = tf.nn.moments(t5, axes = (0x1, 0x2), keepdims = True)
            s5 = tf.sqrt(v5 + epx)
            n5 = (t5 - u5) / s5
            
            t4 = ft[0x4]
            u4, v4 = tf.nn.moments(t4, axes = (0x1, 0x2), keepdims = True)
            s4 = tf.sqrt(v4 + epx)
            n4 = (t4 - u4) / s4
            
            t3 = ft[0x3]
            u3, v3 = tf.nn.moments(t3, axes = (0x1, 0x2), keepdims = True)
            s3 = tf.sqrt(v3 + epx)
            n3 = (t3 - u3) / s3
            
            t2 = ft[0x2]
            u2, v2 = tf.nn.moments(t2, axes = (0x1, 0x2), keepdims = True)
            s2 = tf.sqrt(v2 + epx)
            
            t1 = ft[0x1]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + epx)
            
            t0 = ft[0x0]
            u0, v0 = tf.nn.moments(t0, axes = (0x1, 0x2), keepdims = True)
            s0 = tf.sqrt(v0 + epx)
            
            return (
                (u0, s0),
                (u1, s1),
                (u2, s2),
                (u3, s3, n3),
                (n4,),
                (n5,)
            )
    
    @tf.function(
        input_signature = [
            tf.TensorSpec(_V19NHWC, tf.float32, name = "c1")
        ],
        autograph = False
    )
    def v19GradsX1_0(
            self,
            c1
        ):
            fx0 = self.fx0
            ft1 = self.V19(c1)
    
            eps = self._epx
            
            # Content Correlation Loss
            
            n0, = fx0[0x5]
            t1 = ft1[0x5]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl = tf.math.abs(tf.reduce_mean(n0 * n1, axis = (0x1, 0x2, 0x3)))
            
            n0, = fx0[0x4]
            t1 = ft1[0x4]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl += tf.math.abs(tf.reduce_mean(n0 * n1, axis = (0x1, 0x2, 0x3)))
            
            u0, s0, n0 = fx0[0x3]
            t1 = ft1[0x3]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl += tf.math.abs(tf.reduce_mean(n0 * n1, axis = (0x1, 0x2, 0x3)))
            
            # Style Loss
            
            sl = tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, s0 = fx0[0x2]
            u1, v1 = tf.nn.moments(ft1[0x2], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, s0 = fx0[0x1]
            u1, v1 = tf.nn.moments(ft1[0x1], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, s0 = fx0[0x0]
            u1, v1 = tf.nn.moments(ft1[0x0], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
        
            sl *= self._slw
            
            return tf.gradients(fl + sl, c1), fl

    @tf.function(
        input_signature = [
            tf.TensorSpec(_V19NHWC, tf.float32, name = "c1")
        ],
        autograph = False
    )
    def v19GradsX1_1(
            self,
            c1
        ):
            fx0 = self.fx0
            ft1 = self.V19(c1)
    
            eps = self._epx
            
            # Content Correlation Loss
            n0, = fx0[0x5]
            t1 = ft1[0x5]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl = tf.math.abs(tf.reduce_mean(tf.roll(n0, 0x1, 0x0) * n1, axis = (0x1, 0x2, 0x3)))
            
            n0, = fx0[0x4]
            t1 = ft1[0x4]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl += tf.math.abs(tf.reduce_mean(tf.roll(n0, 0x1, 0x0) * n1, axis = (0x1, 0x2, 0x3)))
            
            u0, s0, n0 = fx0[0x3]
            t1 = ft1[0x3]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl += tf.math.abs(tf.reduce_mean(tf.roll(n0, 0x1, 0x0) * n1, axis = (0x1, 0x2, 0x3)))
            
            # Style Loss
            sl = tf.reduce_mean(
                tf.math.squared_difference(tf.roll(u0, 0x1, 0x0), u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(tf.roll(s0, 0x1, 0x0), s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, s0 = fx0[0x2]
            u1, v1 = tf.nn.moments(ft1[0x2], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(tf.roll(u0, 0x1, 0x0), u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(tf.roll(s0, 0x1, 0x0), s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, s0 = fx0[0x1]
            u1, v1 = tf.nn.moments(ft1[0x1], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(tf.roll(u0, 0x1, 0x0), u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(tf.roll(s0, 0x1, 0x0), s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, s0 = fx0[0x0]
            u1, v1 = tf.nn.moments(ft1[0x0], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(tf.roll(u0, 0x1, 0x0), u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(tf.roll(s0, 0x1, 0x0), s1), axis = (0x1, 0x2, 0x3)
            )
        
            sl *= self._slw
            
            return tf.gradients(fl + sl, c1), fl

    @tf.function(
        input_signature = [
            tf.TensorSpec(_V19NHWC, tf.float32, name = "c1")
        ],
        autograph = False
    )
    def v19GradsX1_2(
            self,
            c1
        ):
            fx0 = self.fx0
            ft1 = self.V19(c1)
    
            eps = self._epx
            
            # Content Correlation Loss
            n0, = fx0[0x5]
            t1 = ft1[0x5]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl = tf.math.abs(tf.reduce_mean(tf.roll(n0, 0x1, 0x0) * n1, axis = (0x1, 0x2, 0x3)))
            
            n0, = fx0[0x4]
            t1 = ft1[0x4]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl += tf.math.abs(tf.reduce_mean(tf.roll(n0, 0x1, 0x0) * n1, axis = (0x1, 0x2, 0x3)))
            
            u0, s0, n0 = fx0[0x3]
            t1 = ft1[0x3]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl += tf.math.abs(tf.reduce_mean(tf.roll(n0, 0x1, 0x0) * n1, axis = (0x1, 0x2, 0x3)))
            
            # Style Loss
            sl = tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, s0 = fx0[0x2]
            u1, v1 = tf.nn.moments(ft1[0x2], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, s0 = fx0[0x1]
            u1, v1 = tf.nn.moments(ft1[0x1], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, s0 = fx0[0x0]
            u1, v1 = tf.nn.moments(ft1[0x0], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(u0, u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(s0, s1), axis = (0x1, 0x2, 0x3)
            )
        
            sl *= self._slw
            
            return tf.gradients(fl + sl, c1), fl
        
    @tf.function(
        input_signature = [
            tf.TensorSpec(_V19NHWC, tf.float32, name = "c1")
        ],
        autograph = False
    )
    def v19GradsX1_3(
            self,
            c1
        ):
            fx0 = self.fx0
            ft1 = self.V19(c1)
    
            eps = self._epx
            
            # Content Correlation Loss
            n0, = fx0[0x5]
            t1 = ft1[0x5]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl = tf.math.abs(tf.reduce_mean(n0 * n1, axis = (0x1, 0x2, 0x3)))
            
            n0, = fx0[0x4]
            t1 = ft1[0x4]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl += tf.math.abs(tf.reduce_mean(n0 * n1, axis = (0x1, 0x2, 0x3)))
            
            u0, s0, n0 = fx0[0x3]
            t1 = ft1[0x3]
            u1, v1 = tf.nn.moments(t1, axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            n1 = (t1 - u1) / s1
            fl += tf.math.abs(tf.reduce_mean(n0 * n1, axis = (0x1, 0x2, 0x3)))
            
            # Style Loss
            sl = tf.reduce_mean(
                tf.math.squared_difference(tf.roll(u0, 0x1, 0x0), u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(tf.roll(s0, 0x1, 0x0), s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, s0 = fx0[0x2]
            u1, v1 = tf.nn.moments(ft1[0x2], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(tf.roll(u0, 0x1, 0x0), u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(tf.roll(s0, 0x1, 0x0), s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, s0 = fx0[0x1]
            u1, v1 = tf.nn.moments(ft1[0x1], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(tf.roll(u0, 0x1, 0x0), u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(tf.roll(s0, 0x1, 0x0), s1), axis = (0x1, 0x2, 0x3)
            )
            
            u0, s0 = fx0[0x0]
            u1, v1 = tf.nn.moments(ft1[0x0], axes = (0x1, 0x2), keepdims = True)
            s1 = tf.sqrt(v1 + eps)
            sl += tf.reduce_mean(
                tf.math.squared_difference(tf.roll(u0, 0x1, 0x0), u1), axis = (0x1, 0x2, 0x3)
            ) + tf.reduce_mean(
                tf.math.squared_difference(tf.roll(s0, 0x1, 0x0), s1), axis = (0x1, 0x2, 0x3)
            )
        
            sl *= self._slw
            
            return tf.gradients(fl + sl, c1), fl
        
    v19GradsX1 = v19GradsX1_1
    
#==================================================
#
#  methods
#
#==================================================

laDSGetPath = lambda paths : [rt + fn for path in paths for rt, ds, fs in os.walk(path) for fn in fs]
laDSGetName = lambda path, pfx : [pfx + fn for rt, ds, fs in os.walk(path) for fn in fs]

def laPreproc(inPath, outPath, seed = 0x0):
    rng = np.random.RandomState(seed)
    if not os.path.exists(outPath) : os.mkdir(outPath)
    inl = len(inPath)
    for rt, ds, fs in os.walk(inPath):
        out = outPath + rt[inl :]
        if not os.path.exists(rt) : os.mkdir(rt)
        for fn in fs:
            # Try decode image
            im0 = cv2.imread(rt + fn)
            
            # Branch if invalid image
            if im0 is None : continue
        
            # Cast
            im0 = im0.astype(np.float32)
            
            # Padding
            s0, s1, s2 = im0.shape
            if s0 > s1:
                im1 = np.zeros((s0, s0, s2))
                a = rng.randint(0x0, s0 - s1)
                im1[:, a : a + s1, :] = im0[:,:,:]
            elif s0 < s1:
                im1 = np.zeros((s1, s1, s2))
                a = rng.randint(0x0, s1 - s0)
                im1[a : a + s1, :, :] = im0[:,:,:]
            else:
                im1 = im0
            print(im1.shape)
            
            # Resize Image
            im = cv2.resize(im1, _V19HW)
            
            # To VGG19
            #im = im[..., ::-0x1] # RGB to BGR (No requried with cv2)
            im -= _V19Mn # BGR means
            
            # I/O to disk
            open(out + fn[: fn.rfind(".")], "wb").write(im.tobytes())

def laDSBuild(ps, n, rt = None):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices(ps)
    if rt:
        ds = ds.map(
            lambda path: tf.reshape(
                tf.io.decode_raw(
                    tf.io.read_file(rt + path),
                    tf.float32
                ),
                _V19HWC
            ),
            num_parallel_calls = AUTOTUNE
        )
    else:
        ds = ds.map(
            lambda path: tf.reshape(
                tf.io.decode_raw(
                    tf.io.read_file(path),
                    tf.float32
                ),
                _V19HWC
            ),
            num_parallel_calls = AUTOTUNE
        )
    ds = ds.batch(n)
    ds = ds.prefetch(buffer_size = AUTOTUNE)
    return ds

def laDSBuildTile(ps, n, rt = None):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices(ps)
    if rt:
        ds = ds.map(
            lambda path: tf.tile(
                tf.reshape(
                        tf.io.decode_raw(
                            tf.io.read_file(rt + path),
                            tf.float32
                        ),
                        _V19BHWC
                ),
                (n, 0x1, 0x1, 0x1)
            ), 
            num_parallel_calls = AUTOTUNE
        )
    else:
        ds = ds.map(
            lambda path: tf.tile(
                tf.reshape(
                        tf.io.decode_raw(
                            tf.io.read_file(path),
                            tf.float32
                        ),
                        _V19BHWC
                ),
                (n, 0x1, 0x1, 0x1)
            ), 
            num_parallel_calls = AUTOTUNE
        )
    ds = ds.prefetch(buffer_size = AUTOTUNE)
    return ds

def laDSBuildEx(ps, n, buf, cnt, rt = None):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices(ps)
    if rt:
        ds = ds.map(
            lambda path: tf.reshape(
                tf.io.decode_raw(
                    tf.io.read_file(rt + path),
                    tf.float32
                ),
                _V19HWC
            ),
            num_parallel_calls = AUTOTUNE
        )
    else:
        ds = ds.map(
            lambda path: tf.reshape(
                tf.io.decode_raw(
                    tf.io.read_file(path),
                    tf.float32
                ),
                _V19HWC
            ),
            num_parallel_calls = AUTOTUNE
        )
    ds = ds.shuffle(buffer_size = buf, reshuffle_each_iteration = True)
    ds = ds.repeat(cnt)
    ds = ds.batch(n)
    ds = ds.prefetch(buffer_size = AUTOTUNE)
    return ds

def laDSBuildTileEx(ps, n, buf, cnt, rt = None):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices(ps)
    if rt:
        ds = ds.map(
            lambda path: tf.tile(
                tf.reshape(
                        tf.io.decode_raw(
                            tf.io.read_file(rt + path),
                            tf.float32
                        ),
                        _V19BHWC
                ),
                (n, 0x1, 0x1, 0x1)
            ), 
            num_parallel_calls = AUTOTUNE
        )
    else:
        ds = ds.map(
            lambda path: tf.tile(
                tf.reshape(
                        tf.io.decode_raw(
                            tf.io.read_file(path),
                            tf.float32
                        ),
                        _V19BHWC
                ),
                (n, 0x1, 0x1, 0x1)
            ), 
            num_parallel_calls = AUTOTUNE
        )
    ds = ds.shuffle(buffer_size = buf, reshuffle_each_iteration = True)
    ds = ds.repeat(cnt)
    ds = ds.prefetch(buffer_size = AUTOTUNE)
    return ds
        
def laAstTrain(obj, crt, srt, epn, stn, bthn, lr, path):
    ast = obj.ast
    trn = ast.train
    sav = ast.save
    cpt = obj.cpt
    spt = obj.spt
    dsc = laDSBuildEx(cpt, bthn, 0x3E8, None, crt)
    dss = laDSBuildEx(spt, bthn, 0x3E8, None, srt)
    
    ctk = dsc.take
    stk = dss.take
    c = next(iter(ctk(0x1)))
    s = next(iter(stk(0x1)))
    bthn = c.shape[0x0]
    fszx = float(bthn << 0x1)
    
    # Train the model
    opti = keras.optimizers.Adam(lr = lr) #, decay=1e-5
    ogds = opti.apply_gradients
    
    ep0 = 0x1
    ckp = tf.train.latest_checkpoint(path)
    if ckp:
        ast.load(ckp)
        ep0 = int(ckp.split('_')[-0x1])
        print("Resume training from epoch %d" % ep0)
        ep0 += 0x1
    
    out, tl, cl, sl = trn(c, s)
    ucl = tf.reduce_mean(cl).numpy()
    usl = tf.reduce_mean(sl).numpy()
    print(f'Input shape: ({c.shape}, {s.shape})')
    print(f'Output shape: {out.shape}')
    print(f'Init. content loss: {ucl:,.2f}, style loss: {usl:,.2f}')
    # mdl.summary()
    
    tvs = ast.trainable_variables
    for ep in range(ep0, epn + 0x1):
        print(f'Epoch {ep}/{epn}')
        for step, c, s in zip(range(stn), ctk(stn), stk(stn)):
            out, gds, cl, sl = trn(c, s)
            ogds(zip(gds, tvs))
            
            ucl = tf.reduce_mean(cl).numpy()
            usl = tf.reduce_mean(sl).numpy()
            
            print(
                f'{step}/{stn} - loss: {ucl + usl:,.2f} - content loss: {ucl:,.2f} - style loss: {usl:,.2f}',
                end='\r'
            ) 
        print()
        
        # Save Model Weight
        sav(os.path.join(path, f'ckpt_{ep}'))
        
        # Plot
        print(
            f'loss: {ucl + usl:,.2f} - content loss: {ucl:,.2f} - style loss: {usl:,.2f}',
            end='\r'
        ) 
        fig, axs = plt.subplots(0x3, bthn, False, False, False)
        fig.set_size_inches(fszx, 6.0)
        fig.set_tight_layout(True)
        tc = c + _tfV19Mn # BGR means
        tc = tc[..., ::-0x1] # BGR to RGB
        ts = s + _tfV19Mn # BGR means
        ts = ts[..., ::-0x1] # BGR to RGB
        tout = out + _tfV19Mn # BGR means
        tout = tout[..., ::-0x1] # BGR to RGB
        for i, ci, si, outi in zip(range(bthn), tc, ts, tout):
            ax = axs[0x0, i]
            img = tf.cast(ci, tf.uint8)
            ax.imshow(img, aspect = "equal")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("content")
            ax = axs[0x1, i]
            img = tf.cast(si, tf.uint8)
            ax.imshow(img, aspect = "equal")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("style")
            ax = axs[0x2, i]
            img = tf.cast(outi, tf.uint8)
            ax.imshow(img, aspect = "equal")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("trans")
        plt.show()
      
def laAstTest(mdl, dsc, dss, path):
    ctk = dsc.take
    stk = dss.take
    c = next(iter(ctk(0x1)))
    s = next(iter(stk(0x1)))
    bthn = c.shape[0x0]
    
    # Load weights
    if path:
        ckp = tf.train.latest_checkpoint(path)
        if ckp : mdl.load(ckp)
    
    out, gds, cl, sl = mdl(c, s)
    
    ucl = tf.reduce_mean(cl).numpy()
    usl = tf.reduce_mean(sl).numpy()
    
    print("Loss : %f\nContent Loss : %f\nStyle Loss : %f" % (ucl + usl, ucl, usl))
    
    fig, axs = plt.subplots(0x3, bthn, False, False, False)
    fig.set_size_inches(float(bthn << 0x1), 6.0)
    fig.set_tight_layout(True)
    tc = c + _tfV19Mn # BGR means
    tc = tc[..., ::-0x1] # BGR to RGB
    ts = s + _tfV19Mn # BGR means
    ts = ts[..., ::-0x1] # BGR to RGB
    tout = out + _tfV19Mn # BGR means
    tout = tout[..., ::-0x1] # BGR to RGB
    for i, ci, si, outi in zip(range(bthn), tc, ts, tout):
        ax = axs[0x0, i]
        img = tf.cast(ci, tf.uint8)
        ax.imshow(img, aspect = "equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("content")
        ax = axs[0x1, i]
        img = tf.cast(si, tf.uint8)
        ax.imshow(img, aspect = "equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("style")
        ax = axs[0x2, i]
        img = tf.cast(outi, tf.uint8)
        ax.imshow(img, aspect = "equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("trans")
    plt.show()

def laDistort(obj, inrt, outrt, seed, shxlo, shxhi, shylo, shyhi, skxlo, skxhi, skylo, skyhi, manrt = None):
    rng = np.random.RandomState(seed)
    if not os.path.exists(outrt) : os.mkdir(outrt)
    
    Mt = np.eye(0x3, dtype = np.float32)
    vhi = _V19Hi.tolist()
    _ckey = obj.cpt
    _skey = obj.spt
    if manrt:
        if not os.path.exists(manrt) : os.mkdir(manrt)
        for _kc in _ckey:
            src = np.fromfile(inrt + _kc, np.float32)
            src.resize(_V19HWC)
            for _ks in _skey:
                Mt[0x0, 0x0] = rng.uniform(skxlo, skxhi)
                Mt[0x1, 0x1] = rng.uniform(skylo, skyhi)
                Mt[0x0, 0x2] = rng.uniform(shxlo, shxhi)
                Mt[0x1, 0x2] = rng.uniform(shylo, shyhi)
                dst = cv2.warpAffine(
                    src,
                    np.matmul(
                            cv2.getRotationMatrix2D(_V19HWhlf, rng.uniform(-60.0, 60.0), 1.0),
                            Mt
                    ),
                    _V19HW,
                    borderMode = cv2.BORDER_CONSTANT,
                    borderValue = vhi
                )
                open(outrt + _kc + _ks, "wb").write(dst.tobytes())
                dst += _V19Mn
                cv2.imwrite(manrt + _kc + _ks + ".png", dst.astype(np.uint8)) # No reverse required
    else:
        for _kc in _ckey:
            src = np.fromfile(inrt + _kc, np.float32)
            src.resize(_V19HWC)
            for _ks in _skey:
                Mt[0x0, 0x0] = rng.uniform(skxlo, skxhi)
                Mt[0x1, 0x1] = rng.uniform(skylo, skyhi)
                Mt[0x0, 0x2] = rng.uniform(shxlo, shxhi)
                Mt[0x1, 0x2] = rng.uniform(shylo, shyhi)
                dst = cv2.warpAffine(
                    src,
                    np.matmul(
                            cv2.getRotationMatrix2D(_V19HWhlf, rng.uniform(-60.0, 60.0), 1.0),
                            Mt
                    ),
                    _V19HW,
                    borderMode = cv2.BORDER_CONSTANT,
                    borderValue = vhi
                )
                open(outrt + _kc + _ks, "wb").write(dst.tobytes())

def laAstTrans(obj, crt, srt, outrt, manrt = None):
    if not os.path.exists(outrt) : os.mkdir(outrt)
    
    mdl = obj.ast.test
    cpt = obj.ppt
    spt = obj.spt
    ccnt = obj.nc
    scnt = obj.ns
    dsc = laDSBuild(cpt, scnt, crt)
    dss = laDSBuild(spt, scnt, srt)
    
    cit = iter(cpt)
    s = next(iter(dss.take(scnt)))
    if manrt:
        if not os.path.exists(manrt) : os.mkdir(manrt)
        for c in iter(dsc.take(ccnt)):
            out = mdl(c, s)
            
            dst = tf.cast(out + _tfV19Mn, dtype='uint8')
            dst = dst[..., ::-0x1] # RGB to BGR
            
            for _out, _dst in zip(out.numpy(), dst):
                name = next(cit)
                open(outrt + name, "wb").write(_out.tobytes())
                tf.io.write_file(manrt + name + ".png", tf.image.encode_png(_dst))
                    
    else:
        for c in iter(dsc.take(ccnt)):
            out = mdl(c, s)
            
            for _out in out.numpy():
                open(outrt + next(cit), "wb").write(_out.tobytes())
    
def laAstTransCoral(obj, crt, srt, outrt, drt = None, manrt = None):
    if not os.path.exists(outrt) : os.mkdir(outrt)
    
    mdl = obj.ast.test
    _epx = obj.ast.eps
    cpt = obj.ppt
    spt = obj.spt
    ccnt = obj.nc
    scnt = obj.ns
    dsc = laDSBuild(cpt, scnt, crt)
    dss = laDSBuild(spt, scnt, srt)
    _s = next(iter(dss.take(scnt)))
    cit = iter(cpt)
    
    cts, fts = laCoralCt(_s, _epx)
    dpt = obj.dpt
    if drt and dpt:
        dsd = laDSBuildEx(dpt, scnt, 0x3E8, None, drt)
        
        if manrt:
            if not os.path.exists(manrt) : os.mkdir(manrt)
            for c, d in zip(dsc.take(ccnt), dsd.take(ccnt)):
                
                s = laCoralEx(d, cts, fts, _epx)
                out = mdl(c, s)
                
                dst = tf.cast(out + _tfV19Mn, dtype='uint8')
                dst = dst[..., ::-0x1] # RGB to BGR
                
                for _out, _dst in zip(out.numpy(), dst):
                    name = next(cit)
                    open(outrt + name, "wb").write(_out.tobytes())
                    tf.io.write_file(manrt + name + ".png", tf.image.encode_png(_dst))
                    
        else:
            for c, d in zip(dsc.take(ccnt), dsd.take(ccnt)):
                
                s = laCoralEx(d, cts, fts, _epx)
                out = mdl(c, s)
                
                for _out in out.numpy():
                    open(outrt + next(cit), "wb").write(_out.tobytes())
    else:
        if manrt:
            if not os.path.exists(manrt) : os.mkdir(manrt)
            for c in iter(dsc.take(ccnt)):
                s = laCoralEx(c, cts, fts, _epx)
                out = mdl(c, s)
                
                dst = tf.cast(out + _tfV19Mn, dtype='uint8')
                dst = dst[..., ::-0x1] # RGB to BGR
                
                for _out, _dst in zip(out.numpy(), dst):
                    name = next(cit)
                    open(outrt + name, "wb").write(_out.tobytes())
                    tf.io.write_file(manrt + name + ".png", tf.image.encode_png(_dst))
                    
        else:
            for c in iter(dsc.take(ccnt)):
                s = laCoralEx(c, cts, fts, _epx)
                out = mdl(c, s)
                
                for _out in out.numpy():
                    open(outrt + next(cit), "wb").write(_out.tobytes())

def laV19FGD(obj, nit, flo, lr, inrt, outrt, manrt = None):
    if not os.path.exists(outrt) : os.mkdir(outrt)
    
    inpt = obj.ppt
    a1 = obj._a1
    a2 = obj._a2
    b1 = obj._b1
    b2 = obj._b2
    epx = obj._epx
    nbth = obj.ns
    incnt = obj.nc
    ds = laDSBuild(inpt, nbth, inrt)
    ZERO = tf.zeros(nbth)
    ONE = tf.ones(nbth)
    itpt = iter(inpt)
    tflo = tf.convert_to_tensor(flo)
    lrv = tf.convert_to_tensor(lr)
    step0 = obj.v19GradsX0
    step1 = obj.v19GradsX1
    if manrt:
        if not os.path.exists(manrt) : os.mkdir(manrt)
        
        plt.figure(figsize = (6.0, 6.0))
        for c0 in iter(ds.take(incnt)):
            
            c1 = tf.identity(c0) # Copy
            
            # Extract Features
            obj.fx0 = step0(
                c0
            )
            
            # Initialize Adam
            b1t = tf.identity(b1)
            b2t = tf.identity(b2)
            gmt = tf.zeros_like(c1)
            gvt = tf.zeros_like(c1)
            
            # Update loop
            for i in range(nit):
                print("step : %d / %d" % (i, nit), end = "\r")
            
                gds, fl = step1(c1)
                
                stop = fl <= tflo
                if tf.reduce_all(stop) : break
            
                # Adam
                _gt = tf.where(stop, ZERO, ONE)[:, None, None, None] * gds[0x0]
            
                lrt = lrv * tf.sqrt(1.0 - b2t) / (1.0 - b1t)
                gmt = b1 * gmt + a1 * _gt
                gvt = b2 * gvt + a2 * tf.square(_gt)
                c1 -= lrt * gmt / tf.sqrt(gvt) + epx
                b1t *= b1
                b2t *= b2
                
                # Clip
                c1 = tf.clip_by_value(c1, _V19Lo, _V19Hi)
            
            ceps = tf.reduce_max(tf.abs(c1 - c0), (0x1, 0x2, 0x3)).numpy()
            
            t1 = tf.cast(c1 + _tfV19Mn, dtype = "uint8")
            t1 = t1[..., ::-0x1] # RGB to BGR
            
            for _c1, _t1 in zip(c1, t1):
                _sfx = next(itpt)
                open(outrt + _sfx, "wb").write(_c1.numpy().tobytes())
                tf.io.write_file(manrt + _sfx + ".png", tf.image.encode_png(_t1))
                
            plt.scatter(ceps, fl.numpy())
            
        plt.savefig(manrt + "eps.png")
        plt.show()
    else:
        for c0 in iter(ds.take(incnt)):
            c1 = tf.identity(c0) # Copy
            
            # Extract Features
            obj.fx0 = step0(
                c0
            )
            
            # Initialize Adam
            b1t = tf.identity(b1)
            b2t = tf.identity(b2)
            gmt = tf.zeros_like(c1)
            gvt = tf.zeros_like(c1)
            
            # Update loop
            for i in range(nit):
                print("step : %d / %d" % (i, nit), end = "\r")
            
                gds, fl = step1(c1)
                
                stop = fl <= tflo
                if tf.reduce_all(stop) : break
            
                # Adam
                _gt = tf.where(stop, ZERO, ONE)[:, None, None, None] * gds[0x0]
            
                lrt = lrv * tf.sqrt(1.0 - b2t) / (1.0 - b1t)
                gmt = b1 * gmt + a1 * _gt
                gvt = b2 * gvt + a2 * tf.square(_gt)
                c1 -= lrt * gmt / tf.sqrt(gvt) + epx
                b1t *= b1
                b2t *= b2
                
                # Clip
                c1 = tf.clip_by_value(c1, _V19Lo, _V19Hi)
            
            for _c1, _t1 in zip(c1, t1):
                open(outrt + next(itpt), "wb").write(_c1.numpy().tobytes())   
    


def laV19FGDC(obj, nit, flo, lr, inrt, outrt, manrt = None):
    if not os.path.exists(outrt) : os.mkdir(outrt)
    
    inpt = obj.ppt
    a1 = obj._a1
    a2 = obj._a2
    b1 = obj._b1
    b2 = obj._b2
    eps = obj._eps
    epx = obj._epx
    nbth = obj.ns
    incnt = obj.nc
    ds = laDSBuild(inpt, nbth, inrt)
    ZERO = tf.zeros(nbth)
    ONE = tf.ones(nbth)
    itpt = iter(inpt)
    tflo = tf.convert_to_tensor(flo)
    lrv = tf.convert_to_tensor(lr)
    step0 = obj.v19GradsX0
    step1 = obj.v19GradsX1
    if manrt:
        if not os.path.exists(manrt) : os.mkdir(manrt)
        
        plt.figure(figsize = (6.0, 6.0))
        for c0 in iter(ds.take(incnt)):
            
            c1 = tf.identity(c0) # Copy
            clo = tf.clip_by_value(c1 - eps, _V19Lo, _V19Hi)
            chi = tf.clip_by_value(c1 + eps, _V19Lo, _V19Hi)
            
            # Extract Features
            obj.fx0 = step0(
                c0
            )
            
            # Initialize Adam
            b1t = tf.identity(b1)
            b2t = tf.identity(b2)
            gmt = tf.zeros_like(c1)
            gvt = tf.zeros_like(c1)
            
            # Update loop
            for i in range(nit):
                print("step : %d / %d" % (i, nit), end = "\r")
            
                gds, fl = step1(c1)
                
                stop = fl <= tflo
                if tf.reduce_all(stop) : break
            
                # Adam
                _gt = tf.where(stop, ZERO, ONE)[:, None, None, None] * gds[0x0]
            
                lrt = lrv * tf.sqrt(1.0 - b2t) / (1.0 - b1t)
                gmt = b1 * gmt + a1 * _gt
                gvt = b2 * gvt + a2 * tf.square(_gt)
                c1 -= lrt * gmt / tf.sqrt(gvt) + epx
                b1t *= b1
                b2t *= b2
                
                # Clip
                c1 = tf.clip_by_value(c1, clo, chi)
            
            ceps = tf.reduce_max(tf.abs(c1 - c0), (0x1, 0x2, 0x3)).numpy()
            
            t1 = tf.cast(c1 + _tfV19Mn, dtype = "uint8")
            t1 = t1[..., ::-0x1] # RGB to BGR
            
            for _c1, _t1 in zip(c1, t1):
                _sfx = next(itpt)
                open(outrt + _sfx, "wb").write(_c1.numpy().tobytes())
                tf.io.write_file(manrt + _sfx + ".png", tf.image.encode_png(_t1))
                
            plt.scatter(ceps, fl.numpy())
            
        plt.savefig(manrt + "eps.png")
        plt.show()
    else:
        for c0 in iter(ds.take(incnt)):
            c1 = tf.identity(c0) # Copy
            clo = tf.clip_by_value(c1 - eps, _V19Lo, _V19Hi)
            chi = tf.clip_by_value(c1 + eps, _V19Lo, _V19Hi)
            
            # Extract Features
            obj.fx0 = step0(
                c0
            )
            
            # Initialize Adam
            b1t = tf.identity(b1)
            b2t = tf.identity(b2)
            gmt = tf.zeros_like(c1)
            gvt = tf.zeros_like(c1)
            
            # Update loop
            for i in range(nit):
                print("step : %d / %d" % (i, nit), end = "\r")
            
                gds, fl = step1(c1)
                
                stop = fl <= tflo
                if tf.reduce_all(stop) : break
            
                # Adam
                _gt = tf.where(stop, ZERO, ONE)[:, None, None, None] * gds[0x0]
            
                lrt = lrv * tf.sqrt(1.0 - b2t) / (1.0 - b1t)
                gmt = b1 * gmt + a1 * _gt
                gvt = b2 * gvt + a2 * tf.square(_gt)
                c1 -= lrt * gmt / tf.sqrt(gvt) + epx
                b1t *= b1
                b2t *= b2
                
                # Clip
                c1 = tf.clip_by_value(c1, clo, chi)
                
            for _c1, _t1 in zip(c1, t1):
                open(outrt + next(itpt), "wb").write(_c1.numpy().tobytes())   
        
            plt.scatter(ceps, fl.numpy())
    
def laV19FBIM(obj, nit, flo, lr, inrt, outrt, manrt = None):
    if not os.path.exists(outrt) : os.mkdir(outrt)
    
    inpt = obj.ppt
    eps = obj._eps
    nbth = obj.ns
    incnt = obj.nc
    ds = laDSBuild(inpt, nbth, inrt)
    ZERO = tf.zeros(nbth)
    ONE = tf.ones(nbth)
    itpt = iter(inpt)
    tflo = tf.convert_to_tensor(flo)
    lrv = tf.convert_to_tensor(lr)
    step0 = obj.v19GradsX0
    step1 = obj.v19GradsX1
    if manrt:
        if not os.path.exists(manrt) : os.mkdir(manrt)
        
        plt.figure(figsize = (6.0, 6.0))
        for c0 in iter(ds.take(incnt)):
            
            c1 = tf.identity(c0) # Copy
            clo = tf.clip_by_value(c1 - eps, _V19Lo, _V19Hi)
            chi = tf.clip_by_value(c1 + eps, _V19Lo, _V19Hi)
            
            # Extract Features
            obj.fx0 = step0(
                c0
            )
            
            # Update loop
            for i in range(nit):
                print("step : %d / %d" % (i, nit), end = "\r")
            
                gds, fl = step1(c1)
                
                stop = fl <= tflo
                if tf.reduce_all(stop) : break
            
                # Adam
                _gt = tf.where(stop, ZERO, ONE)[:, None, None, None] * gds[0x0]

                c1 -= lrv * tf.sign(_gt)
                
                # Clip
                c1 = tf.clip_by_value(c1, clo, chi)
            
            ceps = tf.reduce_max(tf.abs(c1 - c0), (0x1, 0x2, 0x3)).numpy()
            
            t1 = tf.cast(c1 + _tfV19Mn, dtype = "uint8")
            t1 = t1[..., ::-0x1] # RGB to BGR
            
            for _c1, _t1 in zip(c1, t1):
                _sfx = next(itpt)
                open(outrt + _sfx, "wb").write(_c1.numpy().tobytes())
                tf.io.write_file(manrt + _sfx + ".png", tf.image.encode_png(_t1))
                
            plt.scatter(ceps, fl.numpy())
            
        plt.savefig(manrt + "eps.png")
        plt.show()
    else:
        for c0 in iter(ds.take(incnt)):
            c1 = tf.identity(c0) # Copy
            clo = tf.clip_by_value(c1 - eps, _V19Lo, _V19Hi)
            chi = tf.clip_by_value(c1 + eps, _V19Lo, _V19Hi)
            
            # Extract Features
            obj.fx0 = step0(
                c0
            )
            
            # Update loop
            for i in range(nit):
                print("step : %d / %d" % (i, nit), end = "\r")
            
                gds, fl = step1(c1)
                
                stop = fl <= tflo
                if tf.reduce_all(stop) : break
            
                # Adam
                _gt = tf.where(stop, ZERO, ONE)[:, None, None, None] * gds[0x0]

                c1 -= lrv * tf.sign(_gt)
                
                # Clip
                c1 = tf.clip_by_value(c1, clo, chi)
                
            for _c1, _t1 in zip(c1, t1):
                open(outrt + next(itpt), "wb").write(_c1.numpy().tobytes())   
        

def laV19FEval(obj, crt, frt, outrt, emax):
    vf = lambda : 0.25
    ppt = obj.ppt
    l = obj.nc
    n = obj.ns
    t = min(l, emax)
    m = l - 0x1
    dsc = laDSBuildTile(obj.cpt, n, crt)
    dsf0 = laDSBuild(ppt, n, frt)
    dsf1 = laDSBuild(ppt, n, frt)
    ctk = dsc.take
    ftk = dsf1.take
    FL0 = np.zeros((t, m, n), dtype = np.float32)
    FL1 = np.zeros((t, n - 0x1, n), dtype = np.float32)
    FL2 = np.zeros((t, m, n), dtype = np.float32)
    FL3 = np.zeros((t, n), dtype = np.float32)
    loss = obj.v19Loss
    for i0, f0 in zip(range(t), dsf0.take(t)):
        cit = iter(ctk(l))
        fit = iter(ftk(l))
        _FL0 = FL0[i0]
        _FL1 = FL1[i0]
        _FL2 = FL2[i0]
        for i1 in range(i0):
            print(f'Step {i0:>4}/{t} : {i1:>4}/{m}', end='\r')
            tl, fl, sl = loss(f0, tf.random.shuffle(next(fit)))
            _FL0[i1, :] = fl.numpy()
            tl, fl, sl = loss(f0, next(cit))
            _FL2[i1, :] = fl.numpy()
        for i1 in range(0x1, n):
            tl, fl, sl = loss(f0, tf.roll(f0, i1, 0x0))
            _FL1[i1 - 0x1, :] = fl.numpy()
        tl, fl, sl = loss(next(fit), next(cit))
        FL3[i0, :] = fl.numpy()
        for i1 in range(i0, m):
            print(f'Step {i0:>4}/{t} : {i1:>4}/{m}', end='\r')
            tl, fl, sl = loss(f0, tf.random.shuffle(next(fit)))
            _FL0[i1, :] = fl.numpy()
            tl, fl, sl = loss(f0, next(cit))
            _FL2[i1, :] = fl.numpy()
    FL0 = FL0.flatten()
    FL1 = FL1.flatten()
    FL2 = FL2.flatten()
    FL3 = FL3.flatten()
    fig, ax = plt.subplots(0x3, 0x2)
    fig.set_size_inches(12.0, 18.0)
    ax0 = ax[0x0, 0x0]
    ax1 = ax[0x0, 0x1]
    ax2 = ax[0x1, 0x0]
    ax3 = ax[0x1, 0x1]
    ax4 = ax[0x2, 0x0]
    ax5 = ax[0x2, 0x1]
    ax0.set_title("Different Label Gaussian KDE")
    ax0.set_xlabel("Fool / Content Loss (Similarity)")
    ax0.set_ylabel("Density")
    ax1.set_title("Same Label Gaussian KDE")
    ax1.set_xlabel("Fool / Content Loss (Similarity)")
    ax1.set_ylabel("Density")
    ax2.set_title("Gaussian KDE Plot")
    ax2.set_xlabel("Fool / Content Loss (Similarity)")
    ax2.set_ylabel("Density")
    ax3.set_title("Box Plot")
    ax3.set_ylabel("Fool / Content Loss (Similarity)")
    ax4.set_title("Same Label Gaussian KDE")
    ax4.set_xlabel("Fool / Content Loss (Similarity)")
    ax4.set_ylabel("Density")
    ax5.set_title("Box Plot")
    ax5.set_ylabel("Fool / Content Loss (Similarity)")
    fmax = max(np.max(FL0), np.max(FL1), np.max(FL2), np.max(FL3))
    df0 = gaussian_kde(FL0)
    df0.covariance_factor = vf
    xs = np.linspace(0.0, fmax, 0x400)
    ys = df0(xs)
    ax0.plot(xs, ys, "C1")
    ax2.plot(xs, ys, "C1", label = "Diff")
    ax4.plot(xs, ys, "C1", label = "Diff")
    df1 = gaussian_kde(FL1)
    df1.covariance_factor = vf
    ys = df1(xs)
    ax1.plot(xs, ys, "C2")
    ax2.plot(xs, ys, "C2", label = "Same")
    ax4.plot(xs, ys, "C2", label = "Same")
    df2 = gaussian_kde(FL2)
    df2.covariance_factor = vf
    ys = df2(xs)
    ax4.plot(xs, ys, "C1", linestyle = "--", label = "Diff (Orig)")
    df3 = gaussian_kde(FL3)
    df3.covariance_factor = vf
    ys = df3(xs)
    ax4.plot(xs, ys, "C2", linestyle = "--", label = "Same (Orig)")
    ax3.boxplot((FL0, FL1), labels = ("Diff", "Same"))
    ax5.boxplot((FL0, FL1, FL2, FL3), labels = ("Diff", "Same", "Diff (Orig)", "Same (Orig)"))
    ax2.legend(loc = "best")
    ax4.legend(loc = "best")
    fig.tight_layout()
    plt.savefig(outrt + "result.png")
    plt.show()

@tf.function(
    input_signature = [
        tf.TensorSpec(_V19NHWC, tf.float32, name = "c0"),
        tf.TensorSpec(_V19NHWC, tf.float32, name = "c1"),
        tf.TensorSpec(tuple(), tf.float32, name = "epx")
    ],
    autograph = False
)
def laCoral(c0, c1, epx):
    ft0 = tf.reshape(c0, _V19XSC)
    ft1 = tf.reshape(c1, _V19XSC)
    _eye = tf.eye(0x3)
    
    u0, v0 = tf.nn.moments(ft0, axes = (0x1,), keepdims = True)
    s0 = tf.sqrt(v0 + epx)
    n0 = (ft0 - u0) / s0
    ct0 = tf.linalg.matmul(n0, n0, adjoint_a = True) + _eye
    st0, ut0, vt0 = tf.linalg.svd(ct0)
    ct0 = tf.linalg.matmul(tf.linalg.matmul(ut0, tf.linalg.diag(tf.sqrt(st0))), vt0, adjoint_b = True)
    
    u1, v1 = tf.nn.moments(ft1, axes = (0x1,), keepdims = True)
    s1 = tf.sqrt(v1 + epx)
    n1 = (ft1 - u1) / s1
    ct1 = tf.linalg.matmul(n1, n1, adjoint_a = True) + _eye
    st1, ut1, vt1 = tf.linalg.svd(ct1)
    ct1 = tf.linalg.matmul(tf.linalg.matmul(ut1, tf.linalg.diag(tf.sqrt(st1))), vt1, adjoint_b = True)
    
    n2 = tf.linalg.matmul(
        n1,
        tf.linalg.matmul(
            tf.linalg.inv(ct1),
            ct0
        )
    )
    ft2 = n2 * s0 + u0
    ft2 = tf.reshape(ft2, _V19XHWC)
    
    return ft2

@tf.function(
    input_signature = [
        tf.TensorSpec(_V19NHWC, tf.float32, name = "c1"),
        tf.TensorSpec(tuple(), tf.float32, name = "epx")
    ],
    autograph = False
)
def laCoralCt(c1, epx):
    ft1 = tf.reshape(c1 + _tfV19Mn, _V19XSC)
    u1, v1 = tf.nn.moments(ft1, axes = (0x1,), keepdims = True)
    s1 = tf.sqrt(v1 + epx)
    n1 = (ft1 - u1) / s1
    ct1 = tf.linalg.matmul(n1, n1, adjoint_a = True) + tf.eye(0x3)
    st1, ut1, vt1 = tf.linalg.svd(ct1)
    ct1 = tf.linalg.matmul(tf.linalg.matmul(ut1, tf.linalg.diag(tf.sqrt(st1))), vt1, adjoint_b = True)
    ct1 = tf.linalg.inv(ct1)

    return ct1, n1


@tf.function(
    input_signature = [
        tf.TensorSpec(_V19NHWC, tf.float32, name = "c0"),
        tf.TensorSpec((None, 0x3, 0x3), tf.float32, name = "ct1"),
        tf.TensorSpec(_V19NSC, tf.float32, name = "n1"),
        tf.TensorSpec(tuple(), tf.float32, name = "epx")
    ],
    autograph = False
)
def laCoralEx(c0, ct1, n1, epx):
    ft0 = tf.reshape(c0 + _tfV19Mn, _V19XSC)
    
    u0, v0 = tf.nn.moments(ft0, axes = (0x1,), keepdims = True)
    s0 = tf.sqrt(v0 + epx)
    n0 = (ft0 - u0) / s0
    ct0 = tf.linalg.matmul(n0, n0, adjoint_a = True) + tf.eye(0x3)
    st0, ut0, vt0 = tf.linalg.svd(ct0)
    ct0 = tf.linalg.matmul(tf.linalg.matmul(ut0, tf.linalg.diag(tf.sqrt(st0))), vt0, adjoint_b = True)
    
    n2 = tf.linalg.matmul(
        n1,
        tf.linalg.matmul(
            ct1, # Already inversed
            ct0
        )
    )
    ft2 = n2 * s0 + u0
    ft2 = tf.reshape(ft2, _V19XHWC)
    ft2 -= _tfV19Mn
    
    return ft2