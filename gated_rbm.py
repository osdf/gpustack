"""
Gated factored RBM, from
Modeling the joint density of two images under a variety of transformations.
see http://www.cs.toronto.edu/~rfm/morphbm/index.html
"""


import numpy as np


from gnumpy import dot as gdot
from gnumpy import sum as gsum
import gnumpy as gpu


from layer import Layer
from misc import match_table, gaussian, bernoulli


class Gated_RBM(Layer):
    def __init__(self, shape, V=gaussian, params=None, **kwargs):
        """
        """
        self.shape = shape
        self.activ = match_table[H]
        self.p = params
        
        self.V = V
        
        self.factors = shape[1]
        # several helpers
        self.xf_sz = shape[0][0]*shape[1]
        self.xfshape = (shape[0][0], shape[1])

        self.yf_sz = shape[0][1]*shape[1]
        self.yfshape = (shape[0][1], shape[1])
        
        self.fh_sz = shape[1]*shape[2]
        self.fhshape = (shape[1], shape[2])
        
        self._cum_xy = self.xf_sz + self.yf_sz
        self._cum_xyh = self._cum_xy + self.fh_sz
        self.size = self._cum_xyh + shape[2]

    def __repr__(self):
        """
        """
        vrep = str(self.V).split()[1]
        hrep = str(self.H).split()[1]
        rep = "FGRBM-%s-%s-%s-[sparsity--%s:%s]"%(vrep, hrep, self.shape, self.lmbd, self.rho)
        return rep

    def fward(self, params, data):
        pass

    def _fward(self, data):
        pass

    def pt_init(self, init_var=1e-2, init_bias=0., rho=0.5, lmbd=0., l2=0.):
        """
        """
        pt_params = gzeros(self.size + self.shape[0][0] + self.shape[0][1])
        pt_params[:self._cum_xyh] = init_var * gpu.randn(self._cum_xyh) 

        self.pt_score = self.score
        self.pt_grad = self.cd1_3way_grad

        self.l2 = l2
        self.rho = rho
        self.lmbd = lmbd
        self.rho_hat = None

        return pt_params

    def pt_done(self):
        pass

    def score(self,):
        pass

    def cd1_3way_grad(self, params, inputs, **kwargs):
        # suggestion: input generator produces 2tuple of
        # input, one matrix X, one matrix Y
        # shape of parameters: first weights from X and Y to Z,
        # then bias for z, then bias for X,Y -- last two are left
        # away for forward model
        g = gzeros(params.shape)
        x, y = inputs

        weights_xf = params[:self.xf_sz].reshape(self.xfshape)
        weights_yf = params[self.xf_sz:self._cum_xy].reshape(self.yfshape)
        weights_fh = params[self._cum_xy:self._cum_xyh].reshape(self.fhshape)
        bias_h = params[self._cum_xyh:self.size]
        bias_x = params[self.size:-self.shape[0][1]]
        bias_y = params[-self.shape[0][1]:]

        # TODO: renorm weights!
        
        factors_x = gdot(x, weights_xf) 
        factors_y = gdot(y, weights_yf)
        factors = factors_x * factors_y

        h1, h_sampled = bernoulli(factors, wm=weights_fh, bias=bias_h, sampling=True)
        factors_h = gdot(h_sampled, weights_fh.T)

        # TODO: sign!!!
        g[:self.xf_sz] = gdot(x.T, factors_y*factors_h).ravel()
        g[self.xf_sz:self._cum_xy] = gdot(y.T, factors_x*factors_h).ravel()
        g[self._cum_xy:sef._cum_xyh] = gdot(h_sampled.T, factors).ravel()
        d_bx = -x.sum(axis=0)
        d_by = -y.sum(axis=0)
        d_bz = -h1.sum(axis=0)

        # 3way cd
        way = np.random.randn() > 0.5
        if way:
            # reconstruct y (output) first.
            tmp = factors_x * factors_h
            y1, _ = self.V(tmp, wm=weights_yf.T, bias=bias_y)
            factors_y[:] = gdot(y1, weights_yf)
            # then reconstruct x (input).
            tmp = factors_y * factors_h
            x1, _ = self.V(tmp, wm=weights_xf.T, bias=bias_x)
            factors_x[:] = gdot(x1, weights_xf)
        else:
            # reconstruct x (input) first.
            tmp = factors_y * factors_h
            x1, _ = self.V(tmp, wm=weights_xf.T, bias=bias_x)
            factors_x[:] = gdot(x1, weights_xf)
            # then reconstruct y (output).
            tmp = factors_x * factors_h
            y1, _ = self.V(tmp, wm=weights_yf.T, bias=bias_x)
            factors_y[:] = gdot(y1, weights_yf)


        factors[:] = factors_x * factors_y
        h2, _ = bernoulli(factors, wm=weights_hf, bias=bias_h)
        factors_h = gdot(h2, weights_fz.T)

        d_xf = gdot(x1.T, factors_y*factors_z).ravel()
        d_yf = gdot(y1.T factors_x*factors_z).ravel()
        d_fz = gdot(h2.T, factors).ravel()
        d_bx = x1.sum(axis=0)
        d_by = x2.sum(axis=0)
        d_bz = h2.sum(axis=0)
        return g
