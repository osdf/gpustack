"""
"""


import numpy as np
from itertools import izip


from gnumpy import zeros as gzeros

from stack import Stack
from layer import Layer
import chopmunk as munk


class DAE(Stack):
    def __init__(self, ind, schedule):
        super(DAE, self).__init__(ind=ind, schedule=schedule)

    def pretrain(self, schedule):
        super(DAE, self).pretrain(schedule=schedule)

        p = self.params.as_numpy_array()
        del self.params

        # How many parameters in the unrolled model?
        _dec = []
        _enc = [0]
        self.psize = 0
        for layer in self:
            _enc.append(layer.shape[0]*layer.shape[1] + layer.shape[1])
            _dec.append(layer.shape[0]*layer.shape[1] + layer.shape[0])
            self.psize += _enc[-1] + _dec[-1]
        self.enc = np.cumsum(_enc)
        _dec.append(0)
        _dec.reverse()
        self.dec = np.cumsum(_dec) + self.enc[-1]
        print self.enc, self.dec
        # Build up encoder and decoder
        self.encoder = []
        self.params = gzeros(self.psize)
        for layer, (c1, c2) in izip(self, izip(self.enc[:-1], self.enc[1:])):
            self.encoder.append(layer)
            self.params[c1:c2] = p[c1:c2]
            layer.p = self.params[c1:c2]
            print c1, c2
        self.decoder = []
        for layer, (c1, c2) in izip(self[-1::-1], izip(self.dec[:-1], self.dec[1:])):
            print c1, c2
            l = layer.transpose(self.params[c1:c2])
            self.decoder.append(l)
        # Fix missing activations of decoder
        for i, layer in enumerate(self[-2::1]):
            self.decoder[i].activ = layer.activ

        msg = {"msg": "DAE unrolled: %s"%self}
        munk.taggify(self.logging, "pretty").send(msg)

    def score(self, params, inputs, targets, **kwargs):
        data = inputs
        for layer, (c1, c2) in izip(self.encoder, izip(self.enc[:-1], self.enc[1:])):
            data = layer.fward(self.params[c1:c2], data)

        # possible spot for semi supervision?

        for layer, (c1, c2) in izip(self.decoder, izip(self.dec[:-1], self.dec[1:])):
            data = layer.fward(self.params[c1:c2], data)

        return self._score(data, targets)

    def fward(self, inputs, **kwargs):
        data = inputs
        for layer, (c1, c2) in izip(self.encoder, izip(self.enc[:-1], self.enc[1:])):
            data = layer.fward(self.params[c1:c2], data)
        return data

    def grad(self, params, inputs, targets, **kwargs):
        data = inputs
        for layer, (c1, c2) in izip(self.encoder, izip(self.enc[:-1], self.enc[1:])):
            data = layer.fprop(self.params[c1:c2], data)

        # possible spot for semisupervision?

        for layer, (c1, c2) in izip(self.decoder, izip(self.dec[:-1], self.dec[1:])):
            data = layer.fprop(self.params[c1:c2], data)

        _, delta = self.score(data, inputs, error=True)

        g = gzeros(self.psize)
        
        for layer, (c1, c2) in izip(self.decoder[::-1], izip(self.dec[-2::-1], self.dec[:0:-1])):
            delta = layer.bprop(params=params[c1:c2], grad=g[c1:c2], delta=delta)

        # in case: fuse in gradient from semisupervision

        for layer, (c1, c2) in izip(self.encoder[::-1], izip(self.enc[-2::-1], self.enc[:0:-1])):
            delta = layer.bprop(params=params[c1:c2], grad=g[c1:c2], delta=delta)
        return g
