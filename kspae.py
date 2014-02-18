"""
A layer that can be _pretrained_ in
an unsupervised way using a 
k-sparse Autoencoder (kspae).
"""


from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
import gnumpy as gpu


from layer import Layer
from misc import diff_table
from utils import init_SI


class KSpAE(Layer):
    def __init__(self, shape, activ, params=None, **kwargs):
        super(TAE, self).__init__(shape=shape, activ=activ, params=params)

    def __repr__(self):
        activ = str(self.activ).split()[1]
        rep = "TAE-%s-%s"%(activ, self.shape)
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
        idxs = np.argsort(hddn.as_numpy_array(), axis=1)
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
