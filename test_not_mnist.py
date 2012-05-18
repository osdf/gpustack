"""
Experiments on NOT_MNIST.
"""


import numpy as np
from datetime import datetime
from os.path import join


from stack import Stack
import losses
from not_mnist import dataset as ds
import notebook as nb
from utils import log_queue


def run(schedule):
    """
    Laboratory for working with NOT_MNIST.
    """
    print "[LAB_NOT_MNIST] Starting experiment.\n"
    depot = nb.prepare(schedule)

    dset = ds.get_store()
    train = dset["train"]
    test = dset["test"]

    schedule["train"] = (train["inputs"], train["targets"])
    schedule["eval"] = ["train"]
    schedule["logging"] = log_queue(log_to=depot)

    s = Stack(train["inputs"].shape[1], schedule)

    s.pretrain(schedule)
    
    s.train(schedule)
    
    predict = s._predict(test["inputs"])
    print losses.zero_one(predict, test["targets"])
    
    dset.close()
