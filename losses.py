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
        t = 2 * (z > 0) - 1
        return t

    _value = (1 - z * targets)
    maximum = (_value > 0) * _value

    # diff C for unbalance dataset
    # automatically adjust weights inversely proportional to class frequencies
    n, _ = targets.shape
    positive = gpu.sum((targets + 1.) / 2, axis=0)
    negative = n - positive
    inv_ne_freq = float(n) / (negative + 1)
    inv_po_freq = float(n) / (positive + 1)
    class_weight = inv_po_freq * (targets > 0) + inv_ne_freq * (targets < 0)
    # binary hinge loss
    bhl = gpu.sum(maximum ** 2 * class_weight)
    if error:
        err = -2 * targets * maximum * class_weight
        return bhl + addon, err
    else:
        return bhl + addon


def l1svm_mia(z, targets, predict=False, error=False, addon=0):
    """
    l1-SVM for the hinge loss, Multiple independent attributes
    addon, weight
    Note: the targets here are (1, -1)
    """
    if predict:
        # argmax_t(z*t)
        t = 2 * (z > 0) - 1
        return t

    _value = (1 - z * targets)
    indicator = _value > 0
    maximum = indicator * _value
    # diff C for unbalance dataset
    # automatically adjust weights inversely proportional to class frequencies
    n, _ = targets.shape
    positive = gpu.sum((targets + 1.) / 2, axis=0)
    negative = n - positive
    inv_ne_freq = float(n) / (negative + 1)
    inv_po_freq = float(n) / (positive + 1)
    class_weight = inv_po_freq * (targets > 0) + inv_ne_freq * (targets < 0)
    bhl = gpu.sum(maximum * class_weight)
    if error:
        err = -targets * indicator * class_weight
        return bhl + addon, err
    else:
        return bhl + addon


def l2svm_x(z, targets, predict=False, error=False, addon=0):
    """
    l2-SVM for the hinge loss, cross(mutual exclusive)
    addon, weight
    Note: the _targets here are (1, -1)
    and targets are single numbers which indicate the class label
    """
    if predict:
        # argmax(z)
        return gpu.argmax(z, axis=1)

    n, m = z.shape
    # _targets (1, -1)
    _targets = -1 * gpu.ones((n, m))
    # targets only has one label for one data
    _targets[np.arange(n), targets] += 2
    _value = (1 - z * _targets)
    maximum = (_value > 0) * _value
    xhl = gpu.sum(maximum ** 2)
    if error:
        err = -2 * _targets * maximum
        return xhl + addon, err
    else:
        return xhl + addon


def l1svm_x(z, targets, predict=False, error=False, addon=0):
    """
    l1-SVM for the hinge loss, cross(mutual exclusive)
    addon, weight
    Note: the _targets here are (1, -1)
    and targets are single numbers which indicate the class label
    """
    if predict:
        # argmax(z)
        return gpu.argmax(z, axis=1)

    n, m = z.shape
    _targets = -1 * gpu.ones((n, m))
    _targets[np.arange(n), targets] += 2
    _value = (1 - z * _targets)
    indicator = _value > 0
    maximum = indicator * _value
    xhl = gpu.sum(maximum)
    if error:
        err = -_targets * indicator
        return xhl + addon, err
    else:
        return xhl + addon


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
        t = 2 * (z > 0) - 1
        return t

    maximum = np.maximum(1 - z * targets, 0)

    # diff C for unbalance dataset
    # automatically adjust weights inversely proportional to class frequencies
    n, _ = targets.shape
    positive = np.sum((targets + 1.) / 2, axis=0)
    negative = n - positive
    inv_ne_freq = float(n) / (negative + 1)
    inv_po_freq = float(n) / (positive + 1)
    class_weight = inv_po_freq * (targets > 0) + inv_ne_freq * (targets < 0)
    bhl = np.sum(maximum ** 2 * class_weight)
    if error:
        err = -2 * targets * maximum * class_weight
        return bhl + addon, err
    else:
        return bhl + addon


def _l1svm_mia(z, targets, predict=False, error=False, addon=0):
    """
    Note: the targets here are (1, -1)
    """
    if predict:
        # argmax_t(z*t)
        t = 2 * (z > 0) - 1
        return t
    _value = 1 - z * _targets
    indicator = _value > 0
    maximum = indicator * _value

    # diff C for unbalance dataset
    # automatically adjust weights inversely proportional to class frequencies
    n, _ = targets.shape
    positive = np.sum((targets + 1.) / 2, axis=0)
    negative = n - positive
    inv_ne_freq = float(n) / (negative + 1)
    inv_po_freq = float(n) / (positive + 1)
    class_weight = inv_po_freq * (targets > 0) + inv_ne_freq * (targets < 0)
    bhl = np.sum(maximum * class_weight)
    if error:
        err = -targets * indicator * class_weight
        return bhl + addon, err
    else:
        return bhl + addon


def _l2svm_x(z, targets, predict=False, error=False, addon=0):
    """
    Note: the _targets here are (1, -1)
    and targets are single numbers which indicate the class label
    """
    if predict:
        return np.argmax(z, axis=1)

    n, m = z.shape
    _targets = -1 * np.ones((n, m))
    _targets[np.arange(n), targets] = 1
    _value = 1 - z * _targets
    indicator = _value > 0
    maximum = indicator * _value
    xhl = np.sum(maximum ** 2)
    if error:
        err = -2 * _targets * maximum
        return xhl + addon, err
    else:
        return xhl + addon


def _l1svm_x(z, targets, predict=False, error=False, addon=0):
    """
    Note: the _targets here are (1, -1)
    and targets are single numbers which indicate the class label
    """
    if predict:
        return np.argmax(z, axis=1)

    n, m = z.shape
    _targets = -1 * np.ones((n, m))
    _targets[np.arange(n), targets] = 1
    _value = 1 - z * _targets
    indicator = _value > 0
    maximum = indicator * _value
    xhl = np.sum(maximum)
    if error:
        err = -_targets * indicator
        return xhl + addon, err
    else:
        return xhl + addon


loss_table = {
    ssd: _ssd,
    mia: _mia,
    xe: _xe,
    l2svm_mia: _l2svm_mia,
    l1svm_mia: _l1svm_mia,
    l2svm_x: _l2svm_x,
    l1svm_x: _l1svm_x
}
