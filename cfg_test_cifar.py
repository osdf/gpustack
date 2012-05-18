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


lab="test_cifar.py"

l1 = {
    "type": tae.TAE
    ,"units": 512
    ,"activ": gnumpy.logistic
    ,"score": losses.ssd
    ,"opt": {
        "type": GradientDescent
        ,"steprate": 1e-5
        ,"momentum": 0.9
        ,"iargs": (utils.inpt_source,)
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
        ,"iargs": (utils.in_source, utils.trgt_source)
        ,"btsz": 128
        ,"epochs": 10
    }
}

stack = (sm,)

model = {"stack": stack, "whiten":True, "covered":0.99}