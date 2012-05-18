"""
A layer that can be _pretrained_ in
an unsupervised way using a Denoising Tied
AutoEncoder (DTAE).
"""


from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
import gnumpy as gpu


from layer import Layer
from tae import TAE
from misc import diff_table


class DTAE(TAE):
    def __init__(self, shape, activ, params=None, **kwargs):
        super(DTAE, self).__init__(shape=shape, activ=activ, params=params)
        self.noise = kwargs['opt']["noise"]

    def __repr__(self):
        activ = str(self.activ).split()[1]
        score = str(self.score).split()[1]
        rep = "DTAE-%s-%s-%s-%s"%(self.noise, activ, score, self.shape)
        return rep


    def pt_score(self, params, noisy_inpts, targets, l2=0., **kwargs):
        # fprop in tied AE
        hddn = self.activ(gpu.dot(noisy_inpts, params[:self.m_end].reshape(self.shape)) + params[self.m_end:self.m_end+self.shape[1]])
        Z = gdot(hddn, params[:self.m_end].reshape(self.shape).T) + params[-self.shape[0]:]
        sc = self.score(Z, targets)
        return sc

    def pt_grad(self, params, noisy_inpts, targets, l2=0., **kwargs):
        g = gzeros(params.shape)

        hddn = self.activ(gpu.dot(noisy_inpts, params[:self.m_end].reshape(self.shape)) + params[self.m_end:self.m_end+self.shape[1]])
        Z = gdot(hddn, params[:self.m_end].reshape(self.shape).T) + params[-self.shape[0]:]

        _, delta = self.score(Z, targets, error=True)

        g[:self.m_end] = gdot(delta.T, hddn).ravel()
        g[-self.shape[0]:] = delta.sum(axis=0)

        dsc_dha = gdot(delta, params[:self.m_end].reshape(self.shape)) * diff_table[self.activ](hddn)

        g[:self.m_end] += gdot(noisy_inpts.T, dsc_dha).ravel()

        g[self.m_end:-self.shape[0]] = dsc_dha.sum(axis=0)
        # clean up
        del delta
        return g
