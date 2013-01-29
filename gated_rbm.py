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
    def __init__(self, shape, H=bernoulli, V=gaussian, params=None, **kwargs):
        """
        """
        self.shape = shape
        self.activ = match_table[H]
        self.p = params
        self.H = H
        self.V = V
        self.size = (shape[0][0] + shape[0][1])*shape[1] + shape[1] + shape[0][0] + shape[0][1]

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

    def pt_init(self):
        pass

    def pt_done(self):
        pass

    def score(self,):
        pass

    def cd1_3way_grad(self, params, inputs, mf_damp, **kwargs):
        # suggestion: input generator produces 2tuple of
        # input, one matrix X, one matrix Y
        # shape of parameters: first weights from X and Y to Z,
        # then bias for z, then bias for X,Y -- last two are left
        # away for forward model
        g = gzeros(params.shape)

        # normalize parameters with running norm
        # TODO

        weights_xf = params[].reshape(self.xfshape)
        weights_yf = params[].reshape(self.yfshape)
        weights_fz = params[].reshape(self.zfshape)
        bias_x = params[]
        bias_y = params[]
        bias_z = params[]

        factors_x = gdot(inputs, weights_xf) 
        factors_y = gdot(inputs, weights_yf)
        factors = factors_x * factors_y

        h1, h_sampled = bernoulli(factors, wm=weights_fz, bias=bias_h, sampling=True)
        factors_h = gdot(h_sampled, weights_fz.T)

        # TODO: two types of inputs!
        # TODO: check signs!!
        d_xf = gdot(inputs.T, factors_y*factors_h).ravel()
        d_yf = gdot(inputs.T factors_x*factors_h).ravel()
        d_fz = gdot(h_sampled.T, factors).ravel()
        d_bx = -inputs.sum(axis=0)
        d_by = -inputs.sum(axis=0)
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
