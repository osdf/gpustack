"""
All kinds of various kinds...
"""


import gnumpy as gpu
import numpy as np


def Dsigmoid(sx):
    """
    First derivative of the sigmoid(x) (logistic)
    wrt to x *in terms of* sigmoid(x), denoted
    here _sx_.
    """
    return sx*(1-sx)


def Dtanh(tx):
    """
    First derivative of tx = tanh(x)
    wrt to x, *in terms of* tx!
    """
    return 1 - tx**2


def idnty(x):
    """
    Identity activation.
    """
    return x

def Didnty(ix):
    """
    First derivative of ix=idnty(x)
    wrt to x.
    """
    return 1


def _logistic(x):
    """CPU version."""
    return (1 + np.tanh(x/2.))/2.


def bernoulli(data, wm, bias, sampling=False):
    """
    """
    suff = (gpu.dot(data, wm) + bias).logistic()
    if sampling:
        sample = suff.rand() < suff
    else:
        sample = None
    return suff, sample


def _bernoulli(data, wm, bias, sampling=False):
    """CPU version."""
    suff = _logistic(np.dot(data, wm) + bias)
    if sampling:
        sample = np.random.random(suff.shape) < suff
    else:
        sample = None
    return suff, sample


def gaussian(data, wm, bias, sampling=False):
    """
    """
    suff = gpu.dot(data, wm) + bias
    if sampling:
        sample = suff + gpu.randn(suff.shape)
    else:
        sample = None
    return suff, sample


def _gaussian(data, wm, bias, sampling=False):
    """CPU version."""
    suff = np.dot(data, wm) + bias
    if sampling:
        sample = suff + np.random.normal(size=suff.shape)
    else:
        sample = None
    return suff, sample

def nrelu(data, wm, bias, sampling=False):
    """
    A noisy rectified linear unit.
    """
    suff = gpu.dot(data, wm) + bias
    suff *= (suff > 0)
    if sampling:
        sample = suff + (gpu.sqrt(suff.logistic()) * gpu.randn(suff.shape))
        #sample = suff + gpu.randn(suff.shape)
        #sample *= (sample > 0)
    else:
        sample = None
    return suff, sample


def _nrelu(data, wm, bias, sampling=False):
    """CPU version."""
    return None, None


def relu(x):
    return x*(x>0)


def Drelu(x):
    return 1*(x>0)


diff_table = {
        gpu.logistic: Dsigmoid,
        gpu.tanh: Dtanh,
        idnty: Didnty,
        relu: Drelu
        }


cpu_table = {
        gpu.logistic: _logistic,
        gpu.tanh: np.tanh,
        idnty: idnty,
        bernoulli: _bernoulli,
        gaussian: _gaussian,
        nrelu: _nrelu
        }


match_table = {
        bernoulli: gpu.logistic,
        gaussian: idnty,
        nrelu: relu 
        }