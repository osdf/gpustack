class Gated_RBM(Layer):
    def __init__(self)

    
    def pt_init(self):
        pass

    def pt_done(self):
        pass

    def score():
        pass

    def cd1_3way_grad(self, params, inputs, mf_damp, **kwargs):
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

        h1, h_sampled = bernoulli(factors, wm=factors, bias=bias_h, sampling=True)
        factors_h = gdot(h_sampled, weights_fz.T)

        # TODO: two types of inputs!
        # TODO: check signs!!
        d_xf = gdot(inputs.T, factors_y*factors_z).ravel()
        d_yf = gdot(inputs.T factors_x*factors_z).ravel()
        d_fz = gdot(h_sampled.T, factors).ravel()
        d_bx = -inputs.sum(axis=0)
        d_by = -inputs.sum(axis=0)
        d_bz = -h1.sum(axis=0)

        # 3way cd: TODO
        
        # negative phase
        factors_x = gdot(x1, weights_xf) 
        factors_y = gdot(y1, weights_yf)
        factors = factors_x * factors_y

        h2, _ = bernoulli(factors, wm=factors, bias=bias_h)
        factors_h = gdot(h2, weights_fz.T)

        d_xf = gdot(x1.T, factors_y*factors_z).ravel()
        d_yf = gdot(y1.T factors_x*factors_z).ravel()
        d_fz = gdot(h2.T, factors).ravel()
        d_bx = x1.sum(axis=0)
        d_by = x2.sum(axis=0)
        d_bz = h2.sum(axis=0)
        
        return g
