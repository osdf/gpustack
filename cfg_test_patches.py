"""
Testing on CIFAR -- configuration
"""


# if pythonpath doesn't point 
# to the modules needed:
import sys
# add your path(s) here
sys.path.append("..")


import layer, tae
import losses, misc, utils
import gnumpy
from climin.gd import GradientDescent


lab="test_patches.py"

l1 = {
    "type": tae.TAE
    ,"units": 512
    ,"activ": gnumpy.logistic
    ,"score": losses.ssd
    ,"opt": {
        "type": GradientDescent
        ,"steprate": 1e-5
        ,"momentum": 0.9
        ,"iargs": (utils.cycle_inpt,)
        ,"btsz": 128
        ,"epochs": 10
    }
}

sm = {
    "type": layer.Layer
    ,"units": 10
    ,"activ": misc.idnty
    ,"score": losses.xe
    ,"opt": {
        "type": GradientDescent
        ,"steprate": 1e-4
        ,"momentum": 0.9
        ,"iargs": (utils.cycle_inpt, utils.cycle_trgt)
        ,"btsz": 128
        ,"epochs": 10
    }
}

stack = (sm,)

evals = ("evaluate_liberty_64x64.h5", "evaluate_notredame_64x64.h5", "evaluate_yosemite_64x64.h5")
dist = ("L1", "L2", "COSINE")
norms = ("l1", "l2", "id")

model = {"stack": stack, "whiten":True, "covered":0.99,
        "evals": evals, "dist": dist, "norms": norms}
