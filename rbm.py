"""
A layer that can be _pretrained_ as an RBM.
"""


from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
import gnumpy as gpu
import numpy as np

from layer import Layer
from misc import match_table, bernoulli


class RBM(Layer):
    def __init__(self, shape, activ=None, params=None, **kwargs):
        super(RBM, self).__init__(shape=shape, activ=activ, params=params, **kwargs)

    def __repr__(self):
        hrep = str(self.H).split()[1]
        vrep = str(self.V).split()[1]
        rep = "RBM-%s-%s-%s"%(vrep, hrep, self.shape)
        return rep

    def pt_init(self, H=bernoulli, V=bernoulli, init_var=1e-2, init_bias=0., 
            rho=0.5, lmbd=0., l2=0., **kwargs):
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

        self.H = H
        self.V = V 
        self.activ = match_table[H]

        self.pt_score = self.reconstruction
        self.pt_grad = self.grad_cd1

        self.l2 = l2

        self.rho = rho
        self.lmbd = lmbd
        self.rho_hat = None

        return pt_params

    def pt_done(self, pt_params, **kwargs):
        _params = pt_params.as_numpy_array().tolist()
        info = dict({"params": _params, "shape": self.shape})

        self._bias = pt_params[-self.shape[0]:].copy()
        self.p[:] = pt_params[:-self.shape[0]]

        del pt_params

        return info

    def reconstruction(self, params, inputs, **kwargs):
        h1, h_sampled = self.H(inputs, wm=params[:self.m_end].reshape(self.shape), bias=params[self.m_end:-self.shape[0]], sampling=True)
        v2, _ = self.V(h_sampled, wm=params[:self.m_end].reshape(self.shape).T, bias=params[-self.shape[0]:])

        rho_hat = h1.mean()
        rec = ((inputs - v2)**2).sum()

        return np.array([rec, rho_hat])

    def grad_cd1(self, params, inputs, **kwargs):
        g = gzeros(params.shape)

        n, _ = inputs.shape

        h1, h_sampled = self.H(inputs, wm=params[:self.m_end].reshape(self.shape), bias=params[self.m_end:-self.shape[0]], sampling=True)
        v2, _ = self.V(h_sampled, wm=params[:self.m_end].reshape(self.shape).T, bias=params[-self.shape[0]:])
        h2, _ = self.H(v2, wm=params[:self.m_end].reshape(self.shape), bias=params[self.m_end:-self.shape[0]])

        # Note the negative sign: the gradient is 
        # supposed to point into 'wrong' direction,
        # because the used optimizer likes to minimize.
        g[:self.m_end] = (-1./n)*gdot(inputs.T, h1).ravel()
        g[:self.m_end] += (1./n)*gdot(v2.T, h2).ravel()
        g[:self.m_end] += self.l2*params[:self.m_end]

        g[self.m_end:-self.shape[0]] = -h1.mean(axis=0)
        g[self.m_end:-self.shape[0]] += h2.mean(axis=0)

        g[-self.shape[0]:] = -inputs.mean(axis=0)
        g[-self.shape[0]:] += v2.mean(axis=0)

        if self.rho_hat is None:
            self.rho_hat = h1.mean(axis=0)
        else:
            self.rho_hat *= 0.9
            self.rho_hat += 0.1 * h1.mean(axis=0)
        dKL_drho_hat = (self.rho - self.rho_hat)/(self.rho_hat*(1-self.rho_hat))
        h1_1mh1 = h1*(1 - h1)
        g[m_end:m_end+H] -= self.lmbd/n * gsum(h1_1mh1, axis=0) * dKL_drho_hat
        g[:m_end] -= self.lmbd/n * (gdot(inputs.T * prec, h1_1mh1) * dKL_drho_hat).ravel()

        return g
