"""
A layer _pretrained_ as a Sparse Autoencoder.
"""


from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
from gnumpy import logistic
import gnumpy as gpu

from tae import TAE
from losses import bKL
from layer import Layer
from misc import Dsigmoid


class SAE(TAE):
    def __init__(self, shape, rho=0.01, beta=3, params=None, **kwargs):
        self.rho = rho
        self.beta = beta
        super(SAE, self).__init__(shape=shape, activ=logistic, params=params)

    def __repr__(self):
        activ = str(self.activ).split()[1]
        rep = "SAE-%s-%s-%s-%s"%(self.rho, self.beta, activ, self.shape)
        return rep

    def pt_score(self, params, inpts, **kwargs):
        hddn = logistic(gpu.dot(inpts, params[:self.m_end].reshape(self.shape)) + params[self.m_end:self.m_end+self.shape[1]])
        Z = gdot(hddn, params[:self.m_end].reshape(self.shape).T) + params[-self.shape[0]:]

        rho_hat = hddn.mean(axis=0)

        sparsity = self.beta * gpu.sum(bKL(self.rho, rho_hat))
        sc = self.score(Z, inpts, addon=sparsity)
        
        return sc

    def pt_grad(self, params, inpts, **kwargs):
        g = gzeros(params.shape)
        m, _ = inpts.shape

        hddn = logistic(gpu.dot(inpts, params[:self.m_end].reshape(self.shape)) + params[self.m_end:self.m_end+self.shape[1]])
        Z = gdot(hddn, params[:self.m_end].reshape(self.shape).T) + params[-self.shape[0]:]

        rho_hat = hddn.mean(axis=0)
        rho = self.rho
        sparsity = self.beta * gpu.sum(bKL(rho, rho_hat))

        _, delta = self.score(Z, inpts, error=True, addon=sparsity)

        g[:self.m_end] = gdot(delta.T, hddn).ravel()
        g[-self.shape[0]:] = delta.sum(axis=0)

        diff = Dsigmoid(hddn)
        dsparse_dha = -rho/rho_hat + (1-rho)/(1-rho_hat)
        dsc_dha = diff * (gdot(delta, params[:self.m_end].reshape(self.shape)) + self.beta*dsparse_dha/m)

        g[:self.m_end] += gdot(inpts.T, dsc_dha).ravel()

        g[self.m_end:-self.shape[0]] = dsc_dha.sum(axis=0)
        # clean up
        del delta, hddn, Z
        return g
