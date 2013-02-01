"""
Gated factored RBM, from
Modeling the joint density of two images under a variety of transformations.
see http://www.cs.toronto.edu/~rfm/morphbm/index.html
"""


import numpy as np


from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
from gnumpy import sum as gsum
import gnumpy as gpu


from layer import Layer
from misc import match_table, gaussian, bernoulli


class Gated_RBM(Layer):
    def __init__(self, shape, factors, V=gaussian, params=None, **kwargs):
        """
        """
        self.shape = (shape[0], factors, shape[1])
        self.activ = match_table[bernoulli]
        self.p = params
        
        self.V = V
        
        self.factors = factors
        # several helpers
        self.xf_sz = shape[0][0]*factors
        self.xfshape = (shape[0][0], factors)

        self.yf_sz = shape[0][1]*factors
        self.yfshape = (shape[0][1], factors)
        
        self.fh_sz = factors*shape[1]
        self.fhshape = (factors, shape[1])
        
        self._cum_xy = self.xf_sz + self.yf_sz
        self._cum_xyh = self._cum_xy + self.fh_sz
        self.size = self._cum_xyh + shape[1]

        self.avg_nxyf = 0.
        self.avg_nfh = 0.

    def __repr__(self):
        """
        """
        vrep = str(self.V).split()[1]
        rep = "FGRBM-%s-%s-[sparsity--%s:%s]"%(vrep, self.shape, self.lmbd, self.rho)
        return rep

    def fward(self, params, data):
        pass

    def _fward(self, data):
        pass

    def pt_init(self, init_var=1e-2, init_bias=0., avg_nxyf=0.1, avg_nfh=0.1, rho=0.5, lmbd=0., l2=0., **kwargs):
        """
        """
        pt_params = gzeros(self.size + self.shape[0][0] + self.shape[0][1])
        pt_params[:self._cum_xyh] = init_var * gpu.randn(self._cum_xyh) 

        self.pt_score = self.reconstruction
        self.pt_grad = self.cd1_3way_grad

        self.avg_nxyf = avg_nxyf
        self.avg_nfh = avg_nfh

        self.l2 = l2
        self.rho = rho
        self.lmbd = lmbd
        self.rho_hat = None
        
        return pt_params

    def pt_done(self, pt_params, **kwargs):
        _params = pt_params.as_numpy_array().tolist()
        info = dict({"params": _params, "shape": self.shape})

        self.p[:] = pt_params[:self.size]

        return info

    def score(self,):
        pass

    def reconstruction(self, params, inputs, **kwargs):
        """
        """
        x, y = inputs
        n, _ = x.shape

        weights_xf = params[:self.xf_sz].reshape(self.xfshape)
        weights_yf = params[self.xf_sz:self._cum_xy].reshape(self.yfshape)
        weights_fh = params[self._cum_xy:self._cum_xyh].reshape(self.fhshape)
        bias_h = params[self._cum_xyh:self.size]
        bias_x = params[self.size:-self.shape[0][1]]
        bias_y = params[-self.shape[0][1]:]


        factors_x = gdot(x, weights_xf) 
        factors_y = gdot(y, weights_yf)
        factors = factors_x * factors_y

        h, h_sampled = bernoulli(factors, wm=weights_fh, bias=bias_h, sampling=True)
        factors_h = gdot(h_sampled, weights_fh.T)

        # 3way cd
        way = np.random.rand() > 0.5
        if way:
            # reconstruct y (output) first.
            tmp = factors_x * factors_h
            y1, _ = self.V(tmp, wm=weights_yf.T, bias=bias_y, sampling=True)
            factors_y[:] = gdot(y1, weights_yf)
            # then reconstruct x (input).
            tmp = factors_y * factors_h
            x1, _ = self.V(tmp, wm=weights_xf.T, bias=bias_x, sampling=True)
        else:
            # reconstruct x (input) first.
            tmp = factors_y * factors_h
            x1, _ = self.V(tmp, wm=weights_xf.T, bias=bias_x, sampling=True)
            factors_x[:] = gdot(x1, weights_xf)
            # then reconstruct y (output).
            tmp = factors_x * factors_h
            y1, _ = self.V(tmp, wm=weights_yf.T, bias=bias_x, sampling=True)

        rho_hat = h.sum()

        xrec = gsum((x - x1)**2)
        yrec = gsum((y - y1)**2)

        return np.array([xrec, yrec, self.lmbd*rho_hat, self.avg_nxyf, self.avg_nfh])

    def cd1_3way_grad(self, params, inputs, **kwargs):
        SMALL = 1e-7

        g = gzeros(params.shape)
        x, y = inputs
        n, _ = x.shape

        weights_xf = params[:self.xf_sz].reshape(self.xfshape)
        weights_yf = params[self.xf_sz:self._cum_xy].reshape(self.yfshape)
        weights_fh = params[self._cum_xy:self._cum_xyh].reshape(self.fhshape)
        bias_h = params[self._cum_xyh:self.size]
        bias_x = params[self.size:-self.shape[0][1]]
        bias_y = params[-self.shape[0][1]:]

        # normalize weights
        sq_xf = weights_xf * weights_xf
        norm_xf = gpu.sqrt(sq_xf.sum(axis=0)) + SMALL
        sq_yf = weights_yf * weights_yf
        norm_yf = gpu.sqrt(sq_yf.sum(axis=0)) + SMALL
 
        norm_xyf = (norm_xf.mean() + norm_yf.mean())/2.
        self.avg_nxyf *= 0.95
        self.avg_nxyf += 0.05 * norm_xyf
        weights_xf *= (self.avg_nxyf / norm_xf)
        weights_yf *= (self.avg_nxyf / norm_yf)

        sq_fh = weights_fh*weights_fh
        norm_fh = gpu.sqrt(sq_fh.sum(axis=1)) + SMALL
        self.avg_nfh *= 0.95
        self.avg_nfh += 0.05 * norm_fh.mean()
        weights_fh *= (self.avg_nfh / norm_fh[:, gpu.newaxis])
        # normalization done

        factors_x = gdot(x, weights_xf) 
        factors_y = gdot(y, weights_yf)
        factors = factors_x * factors_y

        h, h_sampled = bernoulli(factors, wm=weights_fh, bias=bias_h, sampling=True)
        factors_h = gdot(h_sampled, weights_fh.T)

        g[:self.xf_sz] = -gdot(x.T, factors_y*factors_h).ravel()
        g[self.xf_sz:self._cum_xy] = -gdot(y.T, factors_x*factors_h).ravel()
        g[self._cum_xy:self._cum_xyh] = -gdot(h_sampled.T, factors).ravel()
        g[self._cum_xyh:self.size] = -h.sum(axis=0)
        g[self.size:-self.shape[0][1]] = -x.sum(axis=0) 
        g[-self.shape[0][1]:] = -y.sum(axis=0)

        # 3way cd
        way = np.random.rand() > 0.5
        if way:
            # reconstruct y (output) first.
            tmp = factors_x * factors_h
            y1, _ = self.V(tmp, wm=weights_yf.T, bias=bias_y, sampling=True)
            factors_y[:] = gdot(y1, weights_yf)
            # then reconstruct x (input).
            tmp = factors_y * factors_h
            x1, _ = self.V(tmp, wm=weights_xf.T, bias=bias_x, sampling=True)
            factors_x[:] = gdot(x1, weights_xf)
        else:
            # reconstruct x (input) first.
            tmp = factors_y * factors_h
            x1, _ = self.V(tmp, wm=weights_xf.T, bias=bias_x, sampling=True)
            factors_x[:] = gdot(x1, weights_xf)
            # then reconstruct y (output).
            tmp = factors_x * factors_h
            y1, _ = self.V(tmp, wm=weights_yf.T, bias=bias_x, sampling=True)
            factors_y[:] = gdot(y1, weights_yf)


        factors[:] = factors_x * factors_y
        h1, _ = bernoulli(factors, wm=weights_fh, bias=bias_h)
        factors_h[:] = gdot(h1, weights_fh.T)

        g[:self.xf_sz] += gdot(x1.T, factors_y*factors_h).ravel()
        g[:self.xf_sz] *= 1./n
        
        g[self.xf_sz:self._cum_xy] += gdot(y1.T, factors_x*factors_h).ravel()
        g[self.xf_sz:self._cum_xy] *= 1./n

        g[self._cum_xy:self._cum_xyh] += gdot(h1.T, factors).ravel()
        g[self._cum_xy:self._cum_xyh] *= 1./n

        g[self._cum_xyh:self.size] += h1.sum(axis=0)
        g[self._cum_xyh:self.size] *= 1./n

        g[self.size:-self.shape[0][1]] += x1.sum(axis=0)
        g[self.size:-self.shape[0][1]] *= 1./n

        g[-self.shape[0][1]:] += y1.sum(axis=0)
        g[-self.shape[0][1]:] *= 1./n

        return g
