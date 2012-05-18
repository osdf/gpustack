"""
Testing on MNIST -- configuraiton
"""


import layer, cae
import losses, misc, utils
import gnumpy as gpu
from climin.gd import GradientDescent


lab="test_not_mnist.py"


l1 = {
    "type": cae.CAE
    ,"units": 512
    ,"activ": gpu.logistic
    ,"score": losses.mia
    ,"cae": 3 
    ,"opt": {
        "type": GradientDescent
        ,"steprate": 5e-2
        ,"momentum": 0.#utils.jump(0.5, 0.9, 5*391)
        ,"iargs": (utils.cycle_inpt,)
        ,"btsz": 128
        ,"epochs": 20 * 391
        ,"stop": 391
    }
}

l2 = {
    "type": cae.CAE
    ,"units": 512
    ,"activ": gpu.logistic
    ,"score": losses.mia
    ,"cae": 3 
    ,"opt": {
        "type": GradientDescent
        ,"steprate": 5e-2
        ,"momentum": utils.jump(0.5, 0.9, 5*391)
        ,"iargs": (utils.cycle_inpt,)
        ,"btsz": 128
        ,"epochs": 20 * 391
        ,"stop": 391
    }
}

l3 = {
    "type": cae.CAE
    ,"units": 2048
    ,"activ": gpu.logistic
    ,"score": losses.mia
    ,"cae": 3 
    ,"opt": {
        "type": GradientDescent
        ,"steprate": 5e-2
        ,"momentum": utils.jump(0.5, 0.9, 5*391)
        ,"iargs": (utils.cycle_inpt,)
        ,"btsz": 128
        ,"epochs": 20 * 391
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
        ,"steprate": 1e-1
        ,"momentum": 0.
        ,"iargs": (utils.cycle_inpt, utils.cycle_trgt)
        ,"btsz": 128
        ,"epochs": 10*391
        ,"stop": 391
    }
}

stack = (l1, l2, l3, sm)
#stack = (l1, sm)
model = {
    "stack": stack,
    "score": losses.xe,
    "opt": {
        "type": GradientDescent
        ,"steprate": 5e-2
        ,"momentum": 0.9
        ,"iargs": (utils.cycle_inpt, utils.cycle_trgt)
        ,"btsz": 128
        ,"epochs": 150 * 391
        ,"stop": 391
    }
}
