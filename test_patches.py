"""
Train some models on patchdata.
"""


import cPickle as pickle


from patchdata import dataset as ds
from stack import Stack
from patchdata import evaluate as ev
import notebook as nb
from utils import log_queue
from helpers import helpers


def run(schedule):
    """
    The lab routine for training.
    """
    print "[LAB_PATCHES] Starting experiment."
    depot = nb.prepare(schedule)
    logq = log_queue(log_to=depot)
    schedule["logging"] = logq 

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

    ## did not work at all
    #print "\n[LAB_PATCHES] PCA on training data"
    #comp, s = helpers.pca(cropped["train"]["inputs"], whiten=True)
    #pcaed = ds.at_store(cropped, comp)
    #cropped.close()

    #print "\n[LAB_PATCHES] Stationary ..."
    #stat = ds.stationary_store(pcaed, C=1., cache=True)
    #pcaed.close()
    #train = stat["train"]["inputs"]
    #valid = stat["validation"]["inputs"]


    print "\n[LAB_PATCHES] Stationary ..."
    stat = ds.stationary_store(cropped, C=1., cache=True)
    
    print "\n[LAB_PATCHES] PCA on training data..."
    comp, s = helpers.zca(stat["train"]["inputs"])#, covered=0.99, whiten=True)
    logq.send(dict({"layer": "pca", "params": comp.tolist(), "shape": comp.shape, "s": s.tolist()}))

    pcaed = ds.at_store(stat, comp)
    cropped.close()
    #stat.close()

    train = pcaed["train"]["inputs"]
    valid = pcaed["validation"]["inputs"]

    schedule["train"] = (train, None)
    schedule["valid"] = (valid, None) 
    schedule["eval"] = ["train", "valid"]
 
    print "\n[LAB_PATCHES] Training ..."
    s = Stack(train.shape[1], schedule)
    s.pretrain(schedule)
    
    pcaed.close()

    print "\n\n[LAB_PATCHES] Eval ..."
    ## do NOT confuse with key "eval" above
    evals = schedule["evals"]
    dist = schedule["dist"]
    norms = schedule["norms"]

    rocf = open(depot+".roc.pickle", "w", 0)
    d = dict()
    for e in evals:
        e_store = ds.get_store(e)

        print "\n[LAB_PATCHES] Crop for Evals ..."
        e_crop = ds.crop_store(e_store, 31, 31, 21, 21, cache=True)
        e_store.close()

        #print "\n[LAB_PATCHES] PCAing Evals..."
        #e_at = ds.at_store(e_crop, comp)
       
        #print "\n[LAB_PATCHES] Stationary for Evals..."
        #e_stat =  ds.stationary_store(e_at, C=1.)
        #e_crop.close()
        #rocs = ev.evaluate(e_stat, distances=dist, normalizations=norms
        #        ,latent=s._predict)
 
        print "\n[LAB_PATCHES] Stationary for Evals..."
        e_stat = ds.stationary_store(e_crop, C=1.)
        e_crop.close()
        
        print "\n[LAB_PATCHES] PCAing Evals..."
        e_at = ds.at_store(e_stat, comp)

        rocs = ev.evaluate(e_at, distances=dist,
                normalizations=norms,latent=s._predict)
        d[e] = rocs

        e_stat.close()
        e_at.close()
    rocf = open(depot+".roc.pickle", "wb", 0)
    pickle.dump(d, rocf)
    rocf.close()
