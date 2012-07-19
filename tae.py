"""
A layer that can be _pretrained_ in
an unsupervised way using a Tied
AutoEncoder (TAE).
"""

import numpy as np


from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
import gnumpy as gpu


from layer import Layer
from misc import diff_table


class TAE(Layer):
    def __init__(self, shape, activ, params=None, **kwargs):
        super(TAE, self).__init__(shape=shape, activ=activ, params=params)

    def __repr__(self):
        activ = str(self.activ).split()[1]
        rep = "TAE-%s-%s"%(activ, self.shape)
        return rep

    def pt_init(self, score=None, init_var=1e-2, init_bias=0., **kwargs):
        pt_params = gzeros(self.m_end + self.shape[1] + self.shape[0])
        if init_var is None:
            init_heur = 4*np.sqrt(6./(self.shape[0]+self.shape[1]))
            pt_params[:self.m_end] = gpu.rand(self.m_end)
            pt_params[:self.m_end] *= 2
            pt_params[:self.m_end] -= 1
            pt_params[:self.m_end] *= init_heur
        else:
            pt_params[:self.m_end] = init_var * gpu.randn(self.m_end)

        pt_params[self.m_end:] = init_bias
        self.score = score
        return pt_params

    def pt_done(self, pt_params, **kwargs):
        _params = pt_params.as_numpy_array().tolist()
        info = dict({"params": _params, "shape": self.shape})

        self._bias = pt_params[-self.shape[0]:].copy()
        self.p[:] = pt_params[:-self.shape[0]]

        del pt_params

        return info

    def pt_score(self, params, inpts, **kwargs):
        # fprop in tied AE
        hddn = self.activ(gpu.dot(inpts, params[:self.m_end].reshape(self.shape)) + params[self.m_end:self.m_end+self.shape[1]])
        Z = gdot(hddn, params[:self.m_end].reshape(self.shape).T) + params[-self.shape[0]:]
        sc = self.score(Z, inpts)
        return sc

    def pt_grad(self, params, inpts, **kwargs):
        g = gzeros(params.shape)

        hddn = self.activ(gpu.dot(inpts, params[:self.m_end].reshape(self.shape)) + params[self.m_end:self.m_end+self.shape[1]])
        Z = gdot(hddn, params[:self.m_end].reshape(self.shape).T) + params[-self.shape[0]:]

        _, delta = self.score(Z, inpts, error=True)

        g[:self.m_end] = gdot(delta.T, hddn).ravel()
        g[-self.shape[0]:] = delta.sum(axis=0)

        dsc_dha = gdot(delta, params[:self.m_end].reshape(self.shape)) * diff_table[self.activ](hddn)

        g[:self.m_end] += gdot(inpts.T, dsc_dha).ravel()

        g[self.m_end:-self.shape[0]] = dsc_dha.sum(axis=0)
        # clean up
        del delta
        return g
