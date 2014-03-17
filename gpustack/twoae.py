"""
A layer that can be _pretrained_ in
an unsupervised way using an Untied
AutoEncoder (UAE).
"""

import numpy as np


from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
import gnumpy as gpu


from layer import Layer
from misc import diff_table


class UAE(Layer):
    def __init__(self, shape, activ, params=None, **kwargs):
        self.shape = shape
        self.Tshape = (shape[1], shape[0])
        self.m_end = shape[0]*shape[1]
        self.activ = activ
        self.p = params
        self.size = self.m_end + shape[1]

    def __repr__(self):
        activ = str(self.activ).split()[1]
        rep = "UAE-%s-%s"%(activ, self.shape)
        return rep

    def pt_init(self, score=None, init_var=1e-2, init_bias=0., **kwargs):
        pt_params = gzeros(self.size + self.m_end + self.shape[0])
        if init_var is None:
            init_heur = 4*np.sqrt(6./(self.shape[0]+self.shape[1]))
            pt_params[:self.m_end] = gpu.rand(self.m_end)
            pt_params[:self.m_end] *= 2
            pt_params[:self.m_end] -= 1
            pt_params[:self.m_end] *= init_heur
            
            pt_params[self.size:-self.shape[0]] = gpu.rand(self.m_end)
            pt_params[self.size:-self.shape[0]] *= 2
            pt_params[self.size:-self.shape[0]] -= 1
            pt_params[self.size:-self.shape[0]] *= init_heur
        else: 
            pt_params[:self.m_end] = init_var * gpu.randn(self.m_end)
            pt_params[self.size:-self.shape[0]] = init_var * gpu.randn(self.m_end)
        
        pt_params[self.m_end:self.size] = init_bias
        pt_params[-self.shape[0]:] = init_bias
        self.score = score
        return pt_params

    def pt_done(self, pt_params, **kwargs):
        _params = pt_params.as_numpy_array().tolist()
        info = dict({"params": _params, "shape": self.shape})

        self._bias = pt_params[-self.shape[0]:].copy()
        self.p[:] = pt_params[:self.size]

        del pt_params

        return info

    def pt_score(self, params, inpts, **kwargs):
        hddn = self.activ(gdot(inpts, params[:self.m_end].reshape(self.shape)) + params[self.m_end:self.size])
        Z = gdot(hddn, params[self.size:-self.shape[0]].reshape(self.Tshape)) + params[-self.shape[0]:]

        sc = self.score(Z, inpts)
        
        return sc

    def pt_grad(self, params, inpts, **kwargs):
        g = gzeros(params.shape)
        m, _ = inpts.shape

        hddn = self.activ(gdot(inpts, params[:self.m_end].reshape(self.shape)) + params[self.m_end:self.size])
        Z = gdot(hddn, params[self.size:-self.shape[0]].reshape(self.Tshape)) + params[-self.shape[0]:]

        _, delta = self.score(Z, inpts, error=True)

        g[self.end:-self.shape[0]] = gdot(hddn.T, delta).ravel()
        g[-self.shape[0]:] = delta.sum(axis=0)

        diff = diff_table[self.activ](hddn)
        dsc_dha = diff * gdot(delta, params[:self.m_end].reshape(self.shape))

        g[:self.m_end] = gdot(inpts.T, dsc_dha).ravel()
        g[self.m_end:self.size] = dsc_dha.sum(axis=0)
        # clean up
        del delta, hddn, Z
        return g
