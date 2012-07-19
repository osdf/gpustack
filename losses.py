"""
Different types of losses.
"""


import numpy as np
import gnumpy as gpu
from utils import logsumexp, _logsumexp


def ssd(z, targets, weight=0.5, predict=False, error=False, addon=0):
    """
    """
    if predict:
        return z
    n, m = z.shape
    err = z - targets
    if error:
        # rec. error + first deriv
        return weight*gpu.sum(err**2)/n + addon, 2.*weight*err/n
    else:
        # only return reconstruction error 
        return weight*gpu.sum(err**2)/n + addon


def mia(z, targets, predict=False, error=False, addon=0):
    """
    Multiple independent attributes (i.e. independent
    binary cross entropy errors).

    Feed model output _z_ through logistic to get
    bernoulli distributed variables. 
    """
    bern = gpu.logistic(z)
    if predict:
        return bern
    n, _ = bern.shape
    # loss is binary cross entropy
    # for every output variable
    bce =  -( targets*bern.log() + (1-targets)*(1-bern).log() ).sum(axis=1)
    bce = gpu.mean(bce)
    if error:
        return bce + addon, (bern - targets)/n
    else:
        return bce + addon


def xe(z, targets, predict=False, error=False, addon=0):
    """
    Cross entropy error.
    """
    if predict:
        return gpu.argmax(z, axis=1)

    _xe = z - logsumexp(z, axis=1)
    n, _ = _xe.shape
    xe = -gpu.mean(_xe[np.arange(n), targets])
    if error:
        err = gpu.exp(_xe)
        err[np.arange(n), targets] -= 1
        return xe + addon, err/n
    else:
        return xe + addon


def zero_one(z, targets):
    """
    """
    return (z!=targets).sum()


def bKL(x, y):
    """
    Kullback-Leibler divergence between two
    bernoulli random vectors x and y.
    Note: Not symmetric.
    """
    return x*gpu.log(x/y) + (1-x)*gpu.log((1-x)/(1-y))


def _ssd(z, targets, weight=0.5, predict=False, error=False, addon=0):
    """
    """
    if predict:
        return z
    #
    err = z - targets
    if error:
        # rec. error + first deriv
        return weight*np.sum(err**2) + addon, 2*weight*err
    else:
        # only return reconstruction error 
        return weight*np.sum(err**2) + addon


def _mia(z, targets, predict=False, error=False, addon=0):
    """
    """
    bern = misc._logistic(z)
    if predict:
        return bern
    # loss is binary cross entropy
    # for every output variable
    bce =  -( targets*np.log(bern) + (1-targets)*np.log(1-bern) ).sum()
    if error:
        return bce + addon, z-targets
    else:
        return bce + addon


def _xe(z, targets, predict=False, error=False, addon=0):
    """
    """
    if predict:
        return np.argmax(z, axis=1)

    _xe = z - _logsumexp(z, axis=1)
    n, _ = _xe.shape
    xe = -np.sum(_xe[np.arange(n), targets])
    if error:
        err = np.exp(_xe)
        err[np.arange(n), targets] -= 1
        return xe + addon, err
    else:
        return xe + addon


loss_table = {
        ssd: _ssd,
        mia: _mia,
        xe: _xe
    }
