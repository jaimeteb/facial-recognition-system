import os
import numpy as np
from .facenet_keras import facenet

h5path = 'lib/weights/nn4.small2.v1.h5'

net = facenet()
net.load_weights(h5path)

def get_net():
    return net

def get_features(image):
    img = (image/ 255.).astype(np.float32)
    pre = net.predict(np.expand_dims(img, axis=0))[0]
    ret = {}
    for i, p in enumerate(pre):
        ret[str(i)] = str(p)
    return ret

def get_features_tolist(image):
    img = (image/ 255.).astype(np.float32)
    pre = net.predict(np.expand_dims(img, axis=0))[0]
    return pre.tolist()
