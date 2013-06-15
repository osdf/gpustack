"""
Different types of losses.
"""


import numpy as np
import gnumpy as gpu
from utils import logsumexp, _logsumexp
import misc


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


def rmssd(z, targets, predict=False, error=False, addon=0):
    """
    Root mean sum of squares.
    """
    if predict:
        return z
    n, m = z.shape
    err = z - targets
    per_sample = gpu.sqrt(gpu.sum(err**2, axis=1))

    if error:
        # rec. error + first deriv
        return gpu.sum(per_sample)/n + addon, err/(n*per_sample[:, gpu.newaxis])
    else:
        # only return reconstruction error 
        return gpu.sum(per_sample)/n + addon


def mia(z, targets, predict=False, error=False, addon=0, tiny=1e-10):
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
    bce =  -( targets*(bern+tiny).log() + (1-targets)*(1-bern+tiny).log() ).sum()
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


def l2svm_mia(z, targets, predict=False, error=False, addon=0):
    """
    l2-SVM for the hinge loss, Multiple independent attributes
    addon, weight
    Note: the targets here are (1, -1)
    """
    if predict:
        # argmax_t(z*t)
        t = z > 0
        t = gpu.where(t == 0, -1, t)
        return t

    _maximum = (1 - z * targets)
    _maximum = gpu.where(_maximum < 0, 0, _maximum)
    if error:
        # return gpu.sum(_maximum ** 2, axis=1), \
        #     gpu.sum(-2 * targets * _maximum, axis=1)
        return gpu.sum(_maximum ** 2), -2 * targets * _maximum
    else:
        # return gpu.sum(_maximum ** 2, axis=1)
        return gpu.sum(_maximum ** 2)


def l1svm_mia(z, targets, predict=False, error=False, addon=0):
    """
    l1-SVM for the hinge loss, Multiple independent attributes
    addon, weight
    Note: the targets here are (1, -1)
    """
    if predict:
        # argmax_t(z*t)
        t = z > 0
        t = gpu.where(t == 0, -1, t)
        return t

    _maximum = (1 - z * targets)
    _maximum = gpu.where(_maximum < 0, 0, _maximum)
    _indicator = _maximum > 0
    if error:
        return gpu.sum(_maximum), -targets * _indicator
    else:
        return gpu.sum(_maximum)


def l2svm_x(z, targets, predict=False, error=False, addon=0):
    """
    l2-SVM for the hinge loss, cross(mutual exclusive)
    addon, weight
    Note: the targets here are (1, -1)
    """
    if predict:
        # argmax(z)
        return gpu.argmax(z, axis=1)

    _maximum = (1 - z * targets)
    _maximum = gpu.where(_maximum < 0, 0, _maximum)
    if error:
        # return gpu.sum(_maximum ** 2, axis=1), \
        #     gpu.sum(-2 * targets * _maximum, axis=1)
        return gpu.sum(_maximum ** 2), -2 * targets * _maximum
    else:
        # return gpu.sum(_maximum ** 2, axis=1)
        return gpu.sum(_maximum ** 2)


def l1svm_x(z, targets, predict=False, error=False, addon=0):
    """
    l1-SVM for the hinge loss, cross(mutual exclusive)
    addon, weight
    Note: the targets here are (1, -1)
    """
    if predict:
        # argmax(z)
        return gpu.argmax(z, axis=1)

    _maximum = (1 - z * targets)
    _maximum = gpu.where(_maximum < 0, 0, _maximum)
    _indicator = _maximum > 0
    if error:
        return gpu.sum(_maximum), -targets * _indicator
    else:
        return gpu.sum(_maximum)


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


def _l2svm_mia(z, targets, predict=False, error=False, addon=0):
    """
    Note: the targets here are (1, -1)
    """
    if predict:
        # argmax_t(z*t)
        t = z > 0
        t = np.where(t == 0, -1, t)
        return t

    _maximum = np.maximum(1 - z * targets, 0)
    if error:
        return np.sum(_maximum ** 2), -2 * targets * _maximum
    else:
        return np.sum(_maximum ** 2)


def _l1svm_mia(z, targets, predict=False, error=False, addon=0):
    """
    Note: the targets here are (1, -1)
    """
    if predict:
        # argmax_t(z*t)
        t = z > 0
        t = np.where(t == 0, -1, t)
        return t

    _maximum = np.maximum(1 - z * targets, 0)
    _indicator = _maximum > 0
    if error:
        return np.sum(_maximum), -targets * _indicator
    else:
        return np.sum(_maximum)


def _l2svm_x(z, targets, predict=False, error=False, addon=0):
    """
    Note: the targets here are (1, -1)
    """
    if predict:
        # argmax(z)
        return np.argmax(z, axis=1)

    _maximum = np.maximum(1 - z * targets, 0)
    if error:
        return np.sum(_maximum ** 2), -2 * targets * _maximum
    else:
        return np.sum(_maximum ** 2)


def _l1svm_x(z, targets, predict=False, error=False, addon=0):
    """
    Note: the targets here are (1, -1)
    """
    if predict:
        # argmax(z)
        return np.argmax(z, axis=1)

    _maximum = np.maximum(1 - z * targets, 0)
    _indicator = _maximum > 0
    if error:
        return np.sum(_maximum), -targets * _indicator
    else:
        return np.sum(_maximum)


loss_table = {
    ssd: _ssd,
    mia: _mia,
    xe: _xe,
    l2svm_mia: _l2svm_mia,
    l1svm_mia: _l1svm_mia,
    l2svm_x: _l2svm_x,
    l1svm_x: _l1svm_x
}
