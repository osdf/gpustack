"""
Not-MNIST -- configuraiton
"""


# if pythonpath doesn't point 
# to the modules needed:
#import sys
#sys.path.append("..") # your path goes here

import layer, tae, rbm
import losses, misc, utils
import gnumpy
from climin.gd import GradientDescent


lab="test_not_mnist.py"


l1 = {
    "type": rbm.RBM
    ,"units": 1024
    ,"activ": None
    ,"opt": {
        "type": GradientDescent
        ,"steprate": 1e-3
        ,"momentum": utils.jump(0.5, 0.9, 5*391)
        ,"iargs": (utils.cycle_inpt,)
        ,"btsz": 128
        ,"epochs": 50 * 391
        ,"stop": 391
    }
}

l2 = {
    "type": rbm.RBM
    ,"units": 1024
    ,"activ": None
    ,"opt": {
        "type": GradientDescent
        ,"steprate": 1e-3
        ,"momentum": utils.jump(0.5, 0.9, 5*391)
        ,"iargs": (utils.cycle_inpt,)
        ,"btsz": 128
        ,"epochs": 50 * 391
        ,"stop": 391
    }
}

l3 = {
    "type": rbm.RBM
    ,"units": 1024
    ,"activ": None
    ,"opt": {
        "type": GradientDescent
        ,"steprate": 1e-3
        ,"momentum": utils.jump(0.5, 0.9, 5*391)
        ,"iargs": (utils.cycle_inpt,)
        ,"btsz": 128
        ,"epochs": 50 * 391
        ,"stop": 391
    }
}

sm = {
    "type": layer.Layer
    ,"units": 10
    ,"activ": misc.idnty
    ,"score": losses.xe
    ,"opt": {
        "type": GradientDescent
        ,"steprate": 1e-3
        ,"momentum": 0.
        ,"iargs": (utils.cycle_inpt, utils.cycle_trgt)
        ,"btsz": 128
        ,"epochs": 10 * 391
        ,"stop": 391
    }
}

stack = (l1, l1, l1, l1, sm)
model = {
    "stack": stack,
    "score": losses.xe,
    "opt": {
        "type": GradientDescent
        ,"steprate": 2.5e-3
        ,"momentum": 0.9
        ,"iargs": (utils.cycle_inpt, utils.cycle_trgt)
        ,"btsz": 128
        ,"epochs": 500 * 391
        ,"stop": 391
    }
}
