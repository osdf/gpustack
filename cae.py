"""
A layer _pretrained_ as a Contractive Autoencoder
with sigmoidal hidden units.
"""


from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
from gnumpy import logistic
import gnumpy as gpu

from tae import TAE
from misc import Dsigmoid 


class CAE(TAE):
    def __init__(self, shape, cae, params=None, **kwargs):
        self.cae = cae
        super(CAE, self).__init__(shape=shape, activ=logistic, params=params)

    def __repr__(self):
        activ = str(self.activ).split()[1]
        rep = "CAE-%s-%s"%(activ, self.shape)
        return rep

    def pt_score(self, params, inpts, **kwargs):
        hddn = logistic(gpu.dot(inpts, params[:self.m_end].reshape(self.shape)) + params[self.m_end:self.m_end+self.shape[1]])
        Z = gdot(hddn, params[:self.m_end].reshape(self.shape).T) + params[-self.shape[0]:]

        w = params[:self.m_end].reshape(self.shape)
        cae = gpu.sum(gpu.mean(Dsigmoid(hddn)**2, axis=0) * gpu.sum(w**2, axis=0))
        cae *= self.cae

        sc = self.score(Z, inpts, addon=cae)
        return sc

    def pt_grad(self, params, inpts, **kwargs):
        g = gzeros(params.shape)
        m, _ = inpts.shape

        hddn = logistic(gpu.dot(inpts, params[:self.m_end].reshape(self.shape)) + params[self.m_end:self.m_end+self.shape[1]])
        Z = gdot(hddn, params[:self.m_end].reshape(self.shape).T) + params[-self.shape[0]:]
        
        w = params[:self.m_end].reshape(self.shape)
        cae = gpu.sum(gpu.mean(Dsigmoid(hddn)**2, axis=0) * gpu.sum(w**2, axis=0))
        cae *= self.cae

        _, delta = self.score(Z, inpts, error=True, addon=cae)

        g[:self.m_end] = gdot(delta.T, hddn).ravel()
        g[-self.shape[0]:] = delta.sum(axis=0)

        cae_grad = gpu.mean(Dsigmoid(hddn)**2, axis=0) * w
        cae_grad += (gdot(inpts.T, (Dsigmoid(hddn)**2 * (1-2*hddn)))/m * gpu.sum(w**2, axis=0))
        g[:self.m_end] += self.cae * 2 * cae_grad.ravel()

        dsc_dha = Dsigmoid(hddn) * gdot(delta, params[:self.m_end].reshape(self.shape))

        g[:self.m_end] += gdot(inpts.T, dsc_dha).ravel()

        g[self.m_end:-self.shape[0]] = dsc_dha.sum(axis=0)
        # clean up
        del delta, hddn, Z
        return g
