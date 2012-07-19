"""
A layer that can be _pretrained_ as a Gauss-RBM with
learned variances for the visible units. See:
http://www.cs.toronto.edu/~tang/papers/mr_dbn.pdf
"""


from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
from gnumpy import sum as gsum
import gnumpy as gpu


from layer import Layer
from misc import match_table, cpu_table, bernoulli, gauss


class RBM(Layer):
    def __init__(self, shape, activ=None, params=None, **kwargs):
        """
        """
        super(RBM, self).__init__(shape=shape, activ=activ, params=params)

    def __repr__(self):
        """
        """
        hrep = str(self.H).split()[1]
        rep = "Gauss-RBM-%s-%s"%(hrep, self.shape)
        return rep

    def pt_init(self, H=bernoulli, init_var=1e-2, init_bias=0., **kwargs):
        """
        """
        # 2*self.shape[0]: precision parameters have size shape[0]
        pt_params = gzeros(self.m_end + self.shape[1] + 2*self.shape[0])
        pt_params[:self.m_end] = init_var * gpu.randn(self.m_end)
        pt_params[self.m_end:-self.shape[0]] = init_bias
        pt_params[-self.shape[0]:] = 1.

        self.H = H
        self.activ = match_table[H]

        self.pt_score = self.reconstruction
        self._pt_score = self._reconstruction
        self.pt_grad = self.grad_cd1

        return pt_params

    def pt_done(self, pt_params, **kwargs):
        """
        """
        _params = pt_params.as_numpy_array().tolist()
        info = dict({"params": _params, "shape": self.shape})

        self._bias = pt_params[-2*self.shape[0]:-self.shape[0]].copy()
        self.p[:] = pt_params[:-2*self.shape[0]]

        del pt_params

        return info

    def reconstruction(self, params, inputs, **kwargs):
        """
        """
        m_end = self.m_end
        V = self.shape[0]
        H = self.shape[1]
        wm = params[:m_end].reshape(self.shape)
        prec = params[-V:][:, gpu.newaxis]

        h1, h_sampled = self.H(inputs, wm=prec*wm, bias=params[m_end:m_end+H], sampling=True)
        v2, v_sampled = gauss(h_sampled, wm=(wm/prec).T, bias=params[-(V+H):-V], prec=prec.ravel(), sampling=True)
        return ((inputs - v_sampled)**2).sum()

    def grad_cd1(self, params, inputs, l2=1e-6, **kwargs):
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
        v2, v_sampled = gauss(h_sampled, wm=(wm/prec).T, bias=params[-(V+H):-V], prec=prec.ravel(), sampling=True)
        h2, _ = self.H(v_sampled, wm=prec*wm, bias=params[m_end:m_end+H])

        # Note the negative sign: the gradient is 
        # supposed to point into 'wrong' direction.
        g[:m_end] = (-1./n)*gdot((inputs*prec).T, h1).ravel()
        g[:m_end] += (1./n)*gdot((v_sampled*prec).T, h2).ravel()
        g[:m_end] += l2*params[:self.m_end]

        g[m_end:m_end+H] = -h1.mean(axis=0)
        g[m_end:m_end+H] += h2.mean(axis=0)

        g[-(V+H):-V] = -inputs.mean(axis=0)
        g[-(V+H):-V] += v2.mean(axis=0)
        g[-(V+H):-V] *= prec

        # Gradient for precision (square root of precision)
        g[-V:] = (-1./n)*(gsum(2*prec*inputs*(params[m_end:m_end+V] - inputs), axis=0) + gsum(gdot(inputs.T, h1)*wm, axis=1))
        g[-V:] += (1./n)*(gsum(2*prec*v_sampled*(params[m_end:m_end+V] - v_sampled), axis=0) + gsum(gdot(v_sampled.T, h2)*wm, axis=1))

        return g
