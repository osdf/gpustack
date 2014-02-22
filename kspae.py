"""
A layer that can be _pretrained_ in
an unsupervised way using a 
k-sparse Autoencoder (kspae).
"""


from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
import gnumpy as gpu
import numpy as np


from layer import Layer
from misc import diff_table, idnty
from utils import init_SI


class KSpAE(Layer):
    def __init__(self, shape, k, alpha, activ=idnty, params=None, **kwargs):
        super(KSpAE, self).__init__(shape=shape, activ=activ, params=params)
        self.k = k
        self.ak = alpha * k

    def __repr__(self):
        activ = str(self.activ).split()[1]
        rep = "KSpAE-%s-%s-%s-%s"%(activ, self.shape, self.k, self.ak)
        return rep

    def pt_init(self, score=None, init_var=1e-2, init_bias=0., l2=0., SI=15, **kwargs):
        pt_params = gzeros(self.m_end + self.shape[1] + self.shape[0])
        if init_var is None:
            pt_params[:self.m_end] = gpu.garray(init_SI(self.shape, sparsity=SI)).ravel()
        else:
            pt_params[:self.m_end] = init_var * gpu.randn(self.m_end)

        pt_params[self.m_end:] = init_bias
        self.score = score

        self.l2 = l2
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
        # get indices
        _hddn= hddn.as_numpy_array()
        idxs = np.argsort(_hddn, axis=1)
        _hddn[range(_hddn.shape[0]), idxs[:, self.ak:].T] = 0
        hddn = gpu.garray(_hddn)
        Z = gdot(hddn, params[:self.m_end].reshape(self.shape).T) + params[-self.shape[0]:]

        sc = self.score(Z, inpts)
        return sc

    def pt_grad(self, params, inpts, **kwargs):
        g = gzeros(params.shape)

        hddn = self.activ(gpu.dot(inpts, params[:self.m_end].reshape(self.shape)) + params[self.m_end:self.m_end+self.shape[1]])
        idxs = np.argsort(hddn.as_numpy_array(), axis=1)
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
