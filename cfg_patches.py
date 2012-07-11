"""
Testing on PATCHES -- configuraiton
"""


# if pythonpath doesn't point 
# to the modules needed:
#import sys
#sys.path.append("..") # your path goes here

import layer, rbm, cae, dtae
import losses, misc, utils
import gnumpy as gpu
from climin.gd import GradientDescent


lab="test_patches.py"
btsz = 128
multiplies = 3000 

l1 = {
    "type": rbm.RBM
    ,"units": 2*1024
    ,"V": misc.gaussian
    ,"init_var": None
    ,"opt": {
        "type": GradientDescent
        ,"steprate": 5e-4
        ,"momentum": utils.jump(0.5, 0.9, 10*multiplies)
        ,"iargs": (utils.cycle_inpt,)
        ,"btsz": btsz 
        ,"epochs": 10 * multiplies 
        ,"stop": multiplies
    }
}

evals = ("evaluate_liberty_64x64.h5",) #, "evaluate_notredame_64x64.h5", "evaluate_yosemite_64x64.h5")
dist = ("L1", "L2")
norms = ("l1", "l2", "id")

stack = (l1,)
model = {"samples": btsz*multiplies, "stack": stack, "score": losses.ssd,
        "evals": evals, "dist": dist, "norms": norms}
