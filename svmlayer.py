"""
See: http://www.cs.toronto.edu/~tang/papers/dlsvm.pdf
Deep Learning using Support Vector Machines, Charlie Tang.
"""


import numpy as np


from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
import gnumpy as gpu


from misc import diff_table, cpu_table, str_table, idnty
from layer import layer


class SVMLayer(Layer):
    def __init__(self, shape, C, params=None, dropout=None, **kwargs):
        super(SVMLayer, self).__init__(shape=shape, activ=idnty, params=params, dropout=dropout) 
        self.C = C

    def __repr__(self):
        if self.score is None:
            _score = "no_score"
        else:
            _score = str(self.score).split()[1]
        return "SVMLayer-%s-%s-%s"%(_score, str_table[self.activ], self.shape)

    def fward(self, params, data):
        return gdot(data, params[:self.m_end].reshape(self.shape)) + params[self.m_end:]

    def fprop(self, params, data):
        self.data = data
        self.Z = gdot(data, params[:self.m_end].reshape(self.shape)) + params[self.m_end:]
        return self.Z

    def bprop(self, params, grad, delta):
        # TODO: check next line, is it according 
        # to formula in the paper? delta must be
        # defined correctly!!
        dE_da = delta * diff_table[self.activ](self.Z)
        # gradient of the bias
        grad[self.m_end:] = dE_da.sum(axis=0)
        # gradient of the weights, takes care of weight 'decay' factor (second addend)
        grad[:self.m_end] = gdot(self.data.T, dE_da).ravel() + params[:self.m_end]
        # backpropagate the delta
        delta = gdot(dE_da, params[:self.m_end].reshape(self.shape).T)
        del self.Z
        return delta

    def pt_init(self, score=None, init_var=1e-2, init_bias=0., SI=15, **kwargs):
        if init_var is None:
            self.SI = SI
            self.p[:self.m_end] = gpu.garray(init_SI(self.shape, sparsity=SI)).ravel()
        else:
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
