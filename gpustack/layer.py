"""
Building block for multi layered 
acyclic computation graphs.
"""


import numpy as np


from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
import gnumpy as gpu


from misc import diff_table, cpu_table, str_table
from utils import init_SI


class Layer(object):
    def __init__(self, shape, activ, params=None, dropout=None, **kwargs):
        self.shape = shape
        self.m_end = shape[0] * shape[1]
        self.activ = activ
        self.p = params
        self.size = shape[0] * shape[1] + shape[1]
        self.cpuify = False
        if dropout is not None and dropout > 0:
            assert(0 < dropout < 1), "Dropout needs to be in (0, 1)."
            self.dropout = dropout
            self.fward = self.fward_dropout
            self.fprop = self.fprop_dropout
            self.bprop = self.bprop_dropout
        elif dropout is not None:
            assert(activ is gpu.logistic), "Spikey neurons need sigmoid."
            # negative dropout: want to have spikey neurons
            self.fprop = self.fprop_spike
            self.fward = self.fward_spike

    def __repr__(self):
        if self.score is None:
            _score = "no_score"
        else:
            _score = str(self.score).split()[1]
        return "Layer-%s-%s-%s" % (_score, str_table[self.activ], self.shape)

    def fward(self, params, data):
        return self.activ(gdot(data, params[:self.m_end].reshape(self.shape))\
                + params[self.m_end:])

    def fward_dropout(self, params, data):
        return (1 - self.dropout) * self.activ(gdot(data,\
                params[:self.m_end].reshape(self.shape)) + params[self.m_end:])

    def fward_spike(self, params, data):
        Z = self.activ(gdot(data, params[:self.m_end].reshape(self.shape))\
                + params[self.m_end:])
        spike = Z > gpu.rand(Z.shape)
        return spike

    def fprop(self, params, data):
        self.data = data
        self.Z = self.activ(gdot(data, params[:self.m_end].reshape(self.shape))\
                + params[self.m_end:])
        return self.Z

    def bprop(self, params, grad, delta):
        dE_da = delta * diff_table[self.activ](self.Z)
        # gradient of the bias
        grad[self.m_end:] = dE_da.sum(axis=0)
        # gradient of the weights
        grad[:self.m_end] = gdot(self.data.T, dE_da).ravel()
        # backpropagate the delta
        delta = gdot(dE_da, params[:self.m_end].reshape(self.shape).T)
        del self.Z
        return delta

    def fprop_dropout(self, params, data):
        self.data = data
        self.Z = self.activ(gdot(data, params[:self.m_end].reshape(self.shape)) + params[self.m_end:])
        self.drop = gpu.rand(self.Z.shape) > self.dropout
        self.Z *= self.drop
        return self.Z

    def bprop_dropout(self, params, grad, delta):
        delta *= self.drop
        dE_da = delta * diff_table[self.activ](self.Z)
        # gradient of the bias
        grad[self.m_end:] = dE_da.sum(axis=0)
        # gradient of the weights
        grad[:self.m_end] = gdot(self.data.T, dE_da).ravel()
        # backpropagate the delta
        delta = gdot(dE_da, params[:self.m_end].reshape(self.shape).T)
        del self.Z
        del self.drop
        return delta

    def fprop_spike(self, params, data):
        self.data = data
        self.Z = self.activ(gdot(data, params[:self.m_end].reshape(self.shape)) + params[self.m_end:])
        spike = self.Z > gpu.rand(self.Z.shape)
        return spike

    def transpose(self, params):
        T = Layer(shape=(self.shape[1], self.shape[0]), activ=None, params=params)
        T.pt_init(score=None, init_var=self.init_var, init_bias=0., SI=self.SI)
        return T

    def pt_init(self, score=None, init_var=1e-2, init_bias=0., SI=15, **kwargs):
        if init_var is None:
            self.init_var = None
            self.SI = SI
            self.p[:self.m_end] = gpu.garray(init_SI(self.shape, sparsity=SI)).ravel()
        else:
            self.SI = SI
            self.init_var = init_var
            self.p[:self.m_end] = init_var * gpu.randn(self.m_end)
        self.p[self.m_end:] = init_bias
        self.score = score
        return self.p 

    def pt_done(self, pt_params, **kwargs):
        """
        Do nothing: Pretraining is already working
        on 'real' parameters (see pt_init: self.p is used).
        """
        _params = pt_params.as_numpy_array().tolist()
        info = {"params": _params, "shape": self.shape}
        return info

    def pt_score(self, params, inputs, targets, l2=0, **kwargs):
        Z = self.activ(gpu.dot(inputs, params[:self.m_end].reshape(self.shape)) + params[self.m_end:])
        sc = self.score(Z, targets)
        return sc

    def pt_grad(self, params, inputs, targets, l2=0, **kwargs):
        g = gzeros(params.shape)
        
        Z = self.activ(gpu.dot(inputs, params[:self.m_end].reshape(self.shape)) + params[self.m_end:])
        _, delta = self.score(Z, targets, error=True)

        g[:self.m_end] = gdot(inputs.T, delta).ravel()
        
        g[self.m_end:] = delta.sum(axis=0)
        # clean up
        del delta
        return g

    def _fward(self, data):
        if self.cpuify:
            _params = self._params
        else:
            _params = self.p.as_numpy_array()
        _a = np.dot(data, _params[:self.m_end].reshape(self.shape)) + _params[self.m_end:]
        return cpu_table[self.activ](_a)
