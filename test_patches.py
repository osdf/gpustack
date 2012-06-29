"""
Train some models on CIFAR 10 data.
"""

import numpy as np


from patchdata import dataset as ds
from stack import Stack
from helpers import helpers
from patchdata import evaluate as ev

def run(schedule):
    """
    The lab routine for training.
    """
    print "[LAB_PATCHES] Starting experiment."
    
    print "[LAB_PATCHES] Reading all patches ..."
    patches = ds.get_store()

    #print "[LAB_PATCHES] Selection ..."
    #selection = ds.select(patches, dataset=['liberty'],
    #        index_set=[(100000, 0)])

    #print "[LAB_PATCHES] Cropping ..."
    #cropped = ds.crop_store(selection, 31, 31, 21, 21)

    #print "[LAB_PATHCES] Stationary ..."
    #helpers.stationary(cropped)

    #print "[LAB_PATCHES] Eval ..."
    evals = schedule["evals"]
    dist = schedule["dist"]
    norms = schedule["norms"]

    for e in evals:
        e_store = ds.get_store(e)
        _e = ds.crop_store(e_store, 31, 31, 21, 21)
        helpers.stationary(_e)
        ev.evaluate(_e, distances=dist, normalizations=norms)
        _e.close()
        e_store.close()
    patches.close()
