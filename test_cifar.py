"""
Train some models on CIFAR 10 data.
"""


import sys, os
if os.uname()[0] == "Linux":
    sys.path.append("/home/osendorf/data/cifar10")
else:
    sys.path.append("/Users/osendorf/oc/proj/data/cifar10")


import cifar_ds as ds
import numpy as np
import scipy.linalg as la
from stack import Stack


def run(schedule):
    """
    The lab routine for training.
    """
    print "[LAB_CIFAR] Starting experiment."
    
    print "[LAB_CIFAR] Reading data ..."
    patches = ds.gray()
    
    print "[LAB_CIFAR] Normalizing: Patch wise."
    patches -= np.atleast_2d(patches.mean(axis=1)).T
    patches /= np.atleast_2d(patches.std(axis=1)).T
    
    print "[LAB_CIFAR] PCAing..."
    patches, comp, s = _pca(patches, **schedule)
    print "[LAB_CIFAR] New shape", patches.shape

    schedule["inputs"] = patches

    s = Stack(patches.shape[1], schedule)
    s.pretrain()

    

def _pca(patches, covered=None, whiten=False, **schedule):
    """
    Assume _already_ normalized patches.
    """
    n, d = patches.shape
    # working with covariance + (svd on cov.) is 
    # much faster than svd on patches directly.
    cov = np.dot(patches.T, patches)/n
    u, s, v = la.svd(cov, full_matrices=False)
    if covered is None:
        retained = d
    else:
        total = np.cumsum(s)[-1]
        retained = sum(np.cumsum(s/total) <= covered)
    print covered, whiten
    s = s[0:retained]
    u = u[:,0:retained]
    if whiten:
        comp = np.dot(u, np.diag(1./np.sqrt(s)))
    else:
        comp = u
    rotated = np.dot(patches, comp)
    return rotated, comp, s


def _zca(patches, eps=0.1, **schedule):
    """
    Compute ZCA.
    """
    n, d = patches.shape
    cov = np.dot(patches.T, patches)/n
    u, s, v = la.svd(cov, full_matrices=False)
    print sum(s<eps)
    comp = np.dot(np.dot(u, np.diag(1./np.sqrt(s + eps))), u.T)
    zca = np.dot(patches, comp)
    return zca, comp, s


def _unwhiten(X, comp):
    """
    Inverse process of whitening.
    _comp_ is assumed to be column wise.
    """
    uw = la.pinv2(comp)
    return np.dot(X, uw)
