class M_HPOT(Layer):
    def __init__(self)

    
    def pt_init(self):
        pass

    def pt_done(self):
        pass

    def score():
        pass

    def grad_free_energy_fpcd(self, wrt):
        g = gzeors(params.shape)

        # adjust some parameter values
        # gamma's should never be < 0.1
        check_zero = wrt[gamma] > 0.1
        wrt[gamma] *= check_zero
        wrt[gamma] += (0.1 - check_zero*0.1)
        # actually, take average of gamma's
        wrt[gamma] = wrt[gamma].mean()

        features = gdot(inputs, wrt[vis_fac])
      
        violation = 1+0.5*gdot(feat_sq, wrt[fac_hid])
        cov_h = gamma/violation

        mean_h = (gdot(inputs, wrt[w_mean]) + wrt[bias]).logistic()
        g[vis_fac] = gdot(feat_sq.T, cov_h).ravel()
        g[fac_hid] = gdot(inputs.T, gdot(cov_h, wrt[fac_hid])*features)
        g[gamma] = gsum(violation.log(), axis=1)
        
        g[w_mean]   = -gdot(inputs.T, mean_h)
        g[bias_hid] = -mean_h.sum(axis=0)
        g[bvis]     = -inputs.sum(axis=0)

        # fast weights for pcd

        # negative phase
        # negative samples via HMC
        negative = self.hmc()
        # use fast weights here!
        features = gdot(negative, wrt[vis_fac])
      
        violation = 1+0.5*gdot(feat_sq, wrt[fac_hid])
        cov_h = gamma/violation

        mean_h = (gdot(negative, wrt[w_mean]) + wrt[bias]).logistic()
        g[vis_fac] = gdot(feat_sq.T, cov_h).ravel()
        g[fac_hid] = gdot(negative.T, gdot(cov_h, wrt[fac_hid])*features)
        g[gamma] = gsum(violation.log(), axis=1)
        
        g[w_mean]   += gdot(negative.T, mean_h)
        g[w_mean]   += weightcost * params[w_mean]
        g[bias_hid] += mean_h.sum(axis=0)
        g[bvis]     += negative.sum(axis=0)

        # update fast parameters
        fast_vis_hid = 19./20 * VFf

    def hmc(self, params, inputs):
        _energy = energy_mHPoT()
        g = energy_gradient()
        vel -= 0.5 * hmc_step * g
        neg += hmc_step * vel
        for leap_frogs in xrange(hmc_length - 1):
            g = eneryg_gradient()
            vel -= hmc_step
            neg += hmc_step * vel
        # last half_step
        g = energy_gradient()
        vel -= 0.5 * hmc_step * g
        # energy at new samples
        energy = energy_mHPoT()
        (_energy - energy).exp()

        if avg_reject_rate < target_reject_rate:
            hmc_step = min(hmc_step*1.01, 0.25)
        else:
            hmc_step = max(hmc_step*0.99, 0.001)
        return negatives

    def energy_gradient():
        features = gdot(inputs, wrt[vis_fac])
        feat_sq = 
        cov_h = gamma/(1+0.5*gdot(feat_sq, wrt[fac_hid]))
        mean_h = -(gdot(inputs, wrt[w_mean]) + wrt[bias]).logistic()

