"""
"""


import numpy as np


from stack import Stack
import losses
from mnist import dataset as ds
import notebook as nb
from utils import log_queue


def run(schedule):
    """
    Laboratory for working with MNIST.
    """
    print "[LAB_MNIST] Starting experiment.\n"
    depot = nb.prepare(schedule)

    dset = ds.get_store()
    train = dset["train"]
    valid = dset["validation"]
    test = dset["test"]

    schedule["train"] = (train["inputs"], train["targets"])
    schedule["valid"] = (valid["inputs"], valid["targets"])

    schedule["eval"] = ["train", "valid"]

    schedule["logging"] = log_queue(log_to=depot)

    s = Stack(train["inputs"].shape[1], schedule)

    s.pretrain(schedule)
    
    s.train(schedule)
    
    predict = s._predict(test["inputs"])
    print losses.zero_one(predict, test["targets"])

    dset.close()
