"""
An example for using the framework.
Ties together logging, datasethandling, dispatching
and the training. 

Shown on the exciting MNIST data.
"""


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

    # "train" and "valid" are fixed keywords
    # only use "valid" if a validation set is available
    schedule["train"] = (train["inputs"], train["targets"])
    schedule["valid"] = (valid["inputs"], valid["targets"])

    # "eval" is another fixed keyword.
    # After every epoch, some evaluation is run over
    # the specified datasets.
    schedule["eval"] = ["train", "valid"]

    schedule["logging"] = log_queue(log_to=depot)

    s = Stack(train["inputs"].shape[1], schedule)

    s.pretrain(schedule)
    
    s.train(schedule)
    
    predict = s._predict(test["inputs"])
    print losses.zero_one(predict, test["targets"])

    dset.close()
