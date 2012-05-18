"""
Testing on MNIST -- configuraiton
"""


# if pythonpath doesn't point 
# to the modules needed:
#import sys
#sys.path.append("..") # your path goes here

import layer, tae, rbm
import losses, misc, utils
import gnumpy
from climin.gd import GradientDescent


lab="test_mnist.py"


l1 = {
    "type": rbm.RBM
    ,"units": 512
    ,"activ": None
    ,"opt": {
        "type": GradientDescent
        ,"steprate": 1e-3
        ,"momentum": utils.jump(0.5, 0.9, 5*392)
        ,"iargs": (utils.cycle_inpt,)
        ,"btsz": 128
        ,"epochs": 5
        ,"stop": 1
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
        ,"epochs": 5
        ,"stop": 1
    }
}

stack = (l1, sm)
model = {
    "stack": stack,
    "score": losses.xe,
    "opt": {
        "type": GradientDescent
        ,"steprate": 1e-3
        ,"momentum": 0.9
        ,"iargs": (utils.cycle_inpt, utils.cycle_trgt)
        ,"btsz": 128
        ,"epochs": 1
        ,"stop": 1 
    }
}
