class Gated_RBM(Layer):
    def __init__(self)

    
    def pt_init(self):
        pass

    def pt_done(self):
        pass

    def score():
        pass

    def cd1_grad(self, params, inputs, mf_damp, **kwargs):
        g = gzeros(params.shape)

        factors = gdot(inputs, params[self._vfslice].reshape(self._vfshape)) 
        factors_squared = factors**2
        tmp = gdot(factors_squared, params[self._fhslice].reshape(self._fhshape))
        hidden = sigmoid(tmp + params[self._bhslice])

        # first part of gradients
        _contrib = gdot(hidden, params[self._fhslice].reshape(self._fhshape).T)
        tmp = _contrib*factors
        g[self._vfslice] = -gdot(inputs.T, tmp).ravel()
        g[self._fhslice] = -gdot(factors_squared.T, hidden).ravel()
        g[self._bhslice] = -hidden.sum(axis=0)
        g[self._bvslice] = -inputs.sum(axis=0)

        # inputs no longer needed, overwritten inplace
        data = inputs
        # _contrib from above
        _contrib = gdot(hidden, params[self._fhslice].reshape(self._fhshape).T)
        for i in xrange(mf_iters):
            tmp = gdot(data, params[self._vfslice].reshape(self._vfshape)) * _contrib
            data *= mf_damp
            tmp = gdot(tmp, params[self._vfslice].reshape(self._vfshape))
            data += (1 - mf_damp)*sigmoid(tmp + params[self._bvslice])

        # 1-step hiddens using reconstruction _data_
        factors = gdot(data, params[self._vfslice].reshape(self._vfshape))
        factors_squared = factors**2
        tmp = gdot(factors_squared, params[self._fhslice].reshape(self._fhshape))
        hidden = sigmoid(tmp + params[self._bhslice])
        
        # second part of gradients
        tmp = gdot(hidden, params[self._fhslice].reshape(self._fhshape).T)
        tmp *= factors
        g[self._vfslice] += gdot(data.T, tmp).ravel()
        g[self._fhslice] += gdot(factors_squared.T, hidden).ravel()
        g[self._bhslice] += hidden.sum(axis=0)
        g[self._bvslice] += inputs.sum(axis=0)

        return g
