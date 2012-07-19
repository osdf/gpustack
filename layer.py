"""
Building block for multi layered 
acyclic computation graphs.
"""


import numpy as np
from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
import gnumpy as gpu


from misc import diff_table, cpu_table


class Layer(object):
    def __init__(self, shape, activ, params=None, **kwargs):
        self.shape = shape
        self.m_end = shape[0]*shape[1]
        self.activ = activ
        self.p = params
        self.size = shape[0]*shape[1] + shape[1]

    def __repr__(self):
        _score = str(self.score).split()[1]
        return "Layer-%s-%s"%(_score, self.shape)

    def fward(self, params, data):
        return self.activ(gdot(data, params[:self.m_end].reshape(self.shape)) + params[self.m_end:])

    def fprop(self, params, data):
        self.data = data
        self.Z = self.activ(gdot(data, params[:self.m_end].reshape(self.shape)) + params[self.m_end:])
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

    def pt_init(self, score=None, init_var=1e-2, init_bias=0., **kwargs):
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
        info = dict({"params": _params, "shape": self.shape})
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
        _params = self.p.as_numpy_array()
        _a = np.dot(data, _params[:self.m_end].reshape(self.shape)) + _params[self.m_end:]
        return cpu_table[self.activ](_a)
