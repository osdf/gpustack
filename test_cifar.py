"""
Train on CIFAR 10 data.
"""


import numpy as np


from cifar10 import dataset as ds
from stack import Stack
from helpers import helpers


def run(schedule):
    """
    The lab routine for training.
    """
    print "[LAB_CIFAR] Starting experiment."
    
    print "[LAB_CIFAR] Reading data ..."
    patches = ds.get_store(fname=ds._default_gray)
    
    print "[LAB_CIFAR] Floatify ..."
    patches = helpers.simply_float(patches)

    print "[LAB_CIFAR] Normalizing: Patch wise."
    helpers._stationary(patches['train']['inputs'])
    helpers._stationary(patches['validation']['inputs'])
    helpers._stationary(patches['test']['inputs'])

#    print "[LAB_CIFAR] PCAing..."
#    patches, comp, s = helpers.pca(patches, **schedule)
#    print "[LAB_CIFAR] New shape", patches.shape

#    schedule["inputs"] = patches
#
#    s = Stack(patches.shape[1], schedule)
#    s.pretrain()
    patches.close()
