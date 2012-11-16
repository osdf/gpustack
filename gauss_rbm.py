"""
A layer that can be _pretrained_ as a Gauss-RBM with
learned variances for the visible units. See:
http://www.cs.toronto.edu/~tang/papers/mr_dbn.pdf
"""


import numpy as np
from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
from gnumpy import sum as gsum
import gnumpy as gpu


from layer import Layer
from misc import match_table, cpu_table, bernoulli, gauss


class GAUSS_RBM(Layer):
    def __init__(self, shape, H=bernoulli, params=None, **kwargs):
        """
        """
        activ = match_table[H]
        super(GAUSS_RBM, self).__init__(shape=shape, activ=activ, params=params)
        self.H = H

    def __repr__(self):
        """
        """
        hrep = str(self.H).split()[1]
        rep = "Gauss-RBM-%s-%s-[sparsity--%s:%s]"%(hrep, self.shape, self.lmbd, self.rho)
        return rep

    def pt_init(self, init_var=1e-2, init_bias=0., 
           rho=0.5, lmbd=0., l2=0., **kwargs):
        """
        """
        # 2*self.shape[0]: precision parameters have size shape[0]
        pt_params = gzeros(self.m_end + self.shape[1] + 2*self.shape[0])
        if init_var is None:
            init_heur = 4*np.sqrt(6./(self.shape[0]+self.shape[1]))
            pt_params[:self.m_end] = gpu.rand(self.m_end)
            pt_params[:self.m_end] *= 2
            pt_params[:self.m_end] -= 1
            pt_params[:self.m_end] *= init_heur
        else:
            pt_params[:self.m_end] = init_var * gpu.randn(self.m_end)
        pt_params[:self.m_end] = init_var*gpu.randn(self.m_end)
        pt_params[self.m_end:-self.shape[0]] = init_bias
        pt_params[-self.shape[0]:] = 1.

        self.pt_score = self.reconstruction
        self.pt_grad = self.grad_cd1

        self.l2 = l2

        self.rho = rho
        self.lmbd = lmbd
        self.rho_hat = None

        return pt_params

    def pt_done(self, pt_params, **kwargs):
        """
        """
        _params = pt_params.as_numpy_array().tolist()
        info = {"params": _params, "shape": self.shape}

        self.prep_layer(pt_params)

        del pt_params

        return info

    def prep_layer(self, pt_params):
        """Prepare for layer interpretation.
        """
        prec = pt_params[-self.shape[0]:][:, gpu.newaxis]
        self._bias = pt_params[-2*self.shape[0]:-self.shape[0]].copy()
        self._prec = prec.ravel().copy()
        wm = prec*pt_params[:self.m_end].reshape(self.shape)

        self.p[:self.m_end] = wm.ravel()
        self.p[-self.shape[1]:] = pt_params[self.m_end:self.m_end+self.shape[1]]

    def reload(self, _pt_params):
        """
        """
        if self.p is None:
            self.p = gzeros(self.size)
        pt_params = gpu.as_garray(_pt_params)
        self.prep_layer(pt_params)
        del pt_params

    def reconstruction(self, params, inputs, **kwargs):
        """
        """
        m_end = self.m_end
        V = self.shape[0]
        H = self.shape[1]
        wm = params[:m_end].reshape(self.shape)
        prec = params[-V:][:, gpu.newaxis]

        h1, h_sampled = self.H(inputs, wm=prec*wm, bias=params[m_end:m_end+H], sampling=True)
        v2, v_sampled = gauss(h_sampled, wm=(wm/prec).T, bias=params[-(2*V):-V], prec=prec.T, sampling=True)

        rho_hat = h1.sum()
        rec = gsum((inputs - v_sampled)**2)
        
        return np.array([rec, rho_hat])

    def grad_cd1(self, params, inputs, **kwargs):
        """
        """
        g = gzeros(params.shape)

        n, _ = inputs.shape

        m_end = self.m_end
        V = self.shape[0]
        H = self.shape[1]
        wm = params[:m_end].reshape(self.shape)
        prec = params[-V:][:, gpu.newaxis]

        h1, h_sampled = self.H(inputs, wm=prec*wm, bias=params[m_end:m_end+H], sampling=True)
        v2, v_sampled = gauss(h_sampled, wm=(wm/prec).T, bias=params[-(2*V):-V], prec=prec.T, sampling=True)
        h2, _ = self.H(v_sampled, wm=prec*wm, bias=params[m_end:m_end+H])

        #print h1[0,0], h_sampled[0,0], v2[0,0], v_sampled[0,0]
        # Note the negative sign: the gradient is 
        # supposed to point into 'wrong' direction.
        g[:m_end] = -gdot(inputs.T*prec, h1).ravel()
        g[:m_end] += gdot(v_sampled.T*prec, h2).ravel()
        g[:m_end] *= 1./n
        g[:m_end] += self.l2*params[:m_end]

        g[m_end:m_end+H] = -h1.sum(axis=0)
        g[m_end:m_end+H] += h2.sum(axis=0)
        g[m_end:m_end+H] *= 1./n

        g[-2*V:-V] = -inputs.sum(axis=0)
        g[-2*V:-V] += v_sampled.sum(axis=0)
        g[-2*V:-V] *= 1./n
        g[-2*V:-V] *= (prec**2).T

        #print gsum(g[:m_end]**2), gsum(g[m_end:m_end+H]**2), gsum(g[-2*V:-V]**2)
        # Gradient for square root of precision
        g[-V:] = -gsum(2*prec.T*inputs*(params[-2*V:-V] - inputs/2), axis=0) + gsum(gdot(inputs.T, h1)*wm, axis=1)
        g[-V:] += (gsum(2*prec.T*v_sampled*(params[-2*V:-V] - v_sampled/2), axis=0) + gsum(gdot(v_sampled.T, h2)*wm, axis=1))
        g[-V:] *= 1./n

        #print gsum(g[-V:]**2)
        if self.lmbd > 0.:
            if self.rho_hat is None:
                self.rho_hat = h1.mean(axis=0)
            else:
                self.rho_hat *= 0.9
                self.rho_hat += 0.1 * h1.mean(axis=0)
            dKL_drho_hat = (self.rho - self.rho_hat)/(self.rho_hat*(1-self.rho_hat))
            h1_1mh1 = h1*(1 - h1)
            g[m_end:m_end+H] -= self.lmbd/n * gsum(h1_1mh1, axis=0) * dKL_drho_hat
            g[:m_end] -= self.lmbd/n * (gdot(inputs.T * prec, h1_1mh1) * dKL_drho_hat).ravel()

        #g[:] = -g[:]
        return g
