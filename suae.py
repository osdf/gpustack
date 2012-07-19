"""
A layer _pretrained_ as a Sparse Autoencoder.
"""


from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
from gnumpy import logistic
import gnumpy as gpu

from uae import UAE
from losses import bKL
from misc import Dsigmoid


class SUAE(UAE):
    def __init__(self, shape, rho=0.01, beta=3, params=None, **kwargs):
        super(SUAE, self).__init__(shape=shape, activ=logistic, params=params)
        self.rho = rho
        self.beta = beta
        self.rho_hat = None
        self.rho_hat_grad = None

    def __repr__(self):
        activ = str(self.activ).split()[1]
        score = str(self.score).split()[1]
        rep = "SUAE-%s-%s-%s-%s-%s"%(self.rho, self.beta, activ, score, self.shape)
        return rep

    def pt_score(self, params, inpts, **kwargs):
        hddn = logistic(gdot(inpts, params[:self.m_end].reshape(self.shape)) + params[self.m_end:self.size])
        Z = gdot(hddn, params[self.size:-self.shape[0]].reshape(self.Tshape)) + params[-self.shape[0]:]

        if self.rho_hat == None:
            self.rho_hat = hddn.mean(axis=0)
        else:
            self.rho_hat *= 0.9
            self.rho_hat += 0.1*hddn.mean(axis=0)

        sparsity = self.beta * gpu.sum(bKL(self.rho, self.rho_hat))
        sc = self.score(Z, inpts, addon=sparsity)
        
        return sc

    def pt_grad(self, params, inpts, **kwargs):
        g = gzeros(params.shape)
        m, _ = inpts.shape

        hddn = logistic(gdot(inpts, params[:self.m_end].reshape(self.shape)) + params[self.m_end:self.size])
        Z = gdot(hddn, params[self.size:-self.shape[0]].reshape(self.Tshape)) + params[-self.shape[0]:]

        if self.rho_hat_grad == None:
            self.rho_hat_grad = hddn.mean(axis=0)
        else:
            self.rho_hat_grad *= 0.9
            self.rho_hat_grad += 0.1*hddn.mean(axis=0)

#        rho_hat = hddn.mean(axis=0)
        rho_hat = self.rho_hat_grad
        rho = self.rho
        sparsity = self.beta * gpu.sum(bKL(rho, rho_hat))
 
        _, delta = self.score(Z, inpts, error=True, addon=sparsity)

        g[self.size:-self.shape[0]] = gdot(hddn.T, delta).ravel()
        g[-self.shape[0]:] = delta.sum(axis=0)

        diff = Dsigmoid(hddn)
        dsparse_dha = -rho/rho_hat + (1-rho)/(1-rho_hat)
        dsc_dha = diff * (gdot(delta, params[:self.m_end].reshape(self.shape)) + self.beta*dsparse_dha/m)

        g[:self.m_end] = gdot(inpts.T, dsc_dha).ravel()

        g[self.m_end:self.size] = dsc_dha.sum(axis=0)
        # clean up
        del delta, hddn, Z
        return g
