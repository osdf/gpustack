"""
Train some models on CIFAR 10 data.
"""

from patchdata import dataset as ds
from stack import Stack
from patchdata import evaluate as ev
import notebook as nb
from utils import log_queue


def run(schedule):
    """
    The lab routine for training.
    """
    print "[LAB_PATCHES] Starting experiment."
    depot = nb.prepare(schedule)
    schedule["logging"] = log_queue(log_to=depot)

    print "\n[LAB_PATCHES] Reading all patches ..."
    patches = ds.get_store()

    print "\n[LAB_PATCHES] Selection ..."
    smpl = schedule["samples"]
    selection = ds.select(patches, dataset=['liberty'],
            index_set=[(smpl, 12800)], cache=True)
    patches.close()

    print "\n[LAB_PATCHES] Cropping ..."
    cropped = ds.crop_store(selection, 31, 31, 21, 21, cache=True)
    selection.close()

    print "\n[LAB_PATCHES] Stationary ..."
    stat = ds.stationary_store(cropped, C=10., cache=True)

    train = stat["train"]["inputs"]
    valid = stat["validation"]["inputs"]
    schedule["train"] = (train, train)
    schedule["valid"] = (valid, valid)
    schedule["eval"] = ["train", "valid"]
 
    print "\n[LAB_PATCHES] Training ..."
    s = Stack(train.shape[1], schedule)
    s.pretrain(schedule)
    stat.close()

    print "\n[LAB_PATCHES] Eval ..."
    # do NOT confuse with key "eval" above
    evals = schedule["evals"]
    dist = schedule["dist"]
    norms = schedule["norms"]

    for e in evals:
        e_store = ds.get_store(e)

        print "\n[LAB_PATCHES] Crop for Evals ..."
        e_crop = ds.crop_store(e_store, 31, 31, 21, 21, cache=True)
        e_store.close()

        print "\n[LAB_PATCHES] Stationary for Evals..."
        e_stat =  ds.stationary_store(e_crop, C=10.)
        e_crop.close()

        rocs = ev.evaluate(e_stat, distances=dist,
                normalizations=norms, latent=s._predict)
        e_stat.close()
