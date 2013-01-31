"""
YADAYADA is a simple toolbox for training neural networks with many hidden layers.
It allows to build these deep networks by successivly pretraining these layers in
order to help training the complete deep network. Therefore, the central data structure
is the Stack, a list of layers.
Two aspects are at the center of YADAYADA: (i) It must allow pretraining the single
layers and (ii) it must allow the usage of general optimization algorithms. To facilitate
this, the Stack is managing the complete parameter vector of the deep network. The
various layers only have views into this vector -- they have their own parameters when
in pretraining mode. It only uses its view on the global parameter vector when updating
this global parameter after a finished pretraining session. This explains why every single layer gets always the parameters
passed in when doing forward or backward computations over the full stack.
"""


import numpy as np
from itertools import izip
import h5py
from time import strftime


from gnumpy import zeros as gzeros
import gnumpy as gpu

from losses import loss_table
from utils import prepare_opt, replace_gnumpy_data
import chopmunk as munk


class Stack(list):
    def __init__(self, ind, schedule):
        gpu.seed_rand(seed=None)
        self.logging = schedule["logging"]
        self.psize = 0
        cuts = [0]
        self.stack = schedule["stack"]
        for layer in self.stack:
            ltype = layer["type"]
            units = layer["units"]
            l = ltype.__new__(ltype)
            l.__init__(shape=(ind, units), **layer)
            self.psize += l.size
            self.append(l)
            cuts.append(l.size)
            ind = units
        self.params = gzeros(self.psize)
        self.cuts = np.cumsum(cuts)
        for layer, (c1, c2) in izip(self, izip(self.cuts[:-1], self.cuts[1:])):
            layer.p = self.params[c1:c2]
        self._score = schedule["score"]

    def __repr__(self):
        rep = "|".join([str(l) for l in self])
        return rep

    def pretrain(self, schedule):
        train = [schedule["train"][0], schedule["train"][1]]
        valid = None if not schedule.get("valid") else [schedule["valid"][0], schedule["valid"][1]]

        assert (valid is not None) == ("valid" in schedule["eval"]), "Confusion about validation set!"

        for i, (layer, sched) in enumerate(izip(self, self.stack)):
            pt_params = layer.pt_init(**sched)
            
            opt_schedule = sched["opt"]
            
            pp = {"layer":i, "type":str(layer)}
            munk.taggify(self.logging, "pretty").send(pp)
            log = munk.add_keyvalue(self.logging, "layer", i)
            
            epochs = opt_schedule["epochs"]
            if epochs > 0:
                opt_schedule["f"] = layer.pt_score
                opt_schedule["fprime"] = layer.pt_grad

                opt, evals = prepare_opt(opt_schedule, pt_params, schedule, train, valid)

                stop = opt_schedule["stop"]
                for j, info in enumerate(opt):
                    if (j+1) % stop == 0:
                        for e in evals:
                            info[e] = evals[e](pt_params)
                        info = replace_gnumpy_data(info)
                        log.send(info)
                        
                    if (j+1) == epochs:
                        break
            else:
                pp = {"msg": "NO PRETRAINING of layer %i"%i}
                munk.taggify(self.logging, "pretty").send(pp)

            info = layer.pt_done(pt_params, **sched)
            pt_params = None
            log.send(info)

            # move data forward, save in temporary hdf5
            if i < (len(self) - 1):
                nxt_name = strftime("%Y-%m-%d-%H:%M:%S") + "_L" + str(i+1) + "_TMP.h5"
                nxt = h5py.File(nxt_name)
                pp = {"msg": "Take care of temporary " + nxt_name}
                munk.taggify(self.logging, "pretty").send(pp)
                if valid:
                    valid[0] = self.next_hdf5(layer, valid[0], "validation", nxt, chunk=512)
                train[0] = self.next_hdf5(layer, train[0], "train", nxt, chunk=512)

    def train(self, schedule):
        train = [schedule["train"][0], schedule["train"][1]]
        valid = None if not schedule.get("valid") else [schedule["valid"][0], schedule["valid"][1]]

        assert (valid is not None) == ("valid" in schedule["eval"]), "Confusion about validation set!"

        opt_schedule = schedule["opt"]
        
        pp = {"type" : str(self)}
        munk.taggify(self.logging, "pretty").send(pp)
        log = munk.add_keyvalue(self.logging, "layer", "Stack")
       
        epochs = opt_schedule["epochs"]
        if epochs > 0:
            opt_schedule["f"] = self.score
            opt_schedule["fprime"] = self.grad

            opt, evals = prepare_opt(opt_schedule, self.params, schedule, train, valid)

            stop = opt_schedule["stop"]
            for i, info in enumerate(opt):
                if (i+1) % stop == 0:
                    for e in evals:
                        info[e] = evals[e](self.params)
                    info = replace_gnumpy_data(info)
                    log.send(info)

                if i+1 == epochs:
                    break
        else:
            pp = {"msg": "NO FINETUNING of stack"}
            munk.taggify(self.logging, "pretty").send(pp)

    def score(self, params, inputs, targets, **kwargs):
        data = inputs
        for layer, (c1, c2) in izip(self, izip(self.cuts[:-1], self.cuts[1:])):
            data = layer.fward(self.params[c1:c2], data)
        return self._score(data, targets)

    def grad(self, params, inputs, targets, **kwargs):
        data = inputs
        for layer, (c1, c2) in izip(self, izip(self.cuts[:-1], self.cuts[1:])):
            data = layer.fprop(self.params[c1:c2], data)

        _, delta = self._score(data, targets, error=True)

        g = gzeros(self.psize)
        for layer, (c1, c2) in izip(self[::-1], izip(self.cuts[-2::-1], self.cuts[:0:-1])):
            delta = layer.bprop(params=params[c1:c2], grad=g[c1:c2], delta=delta)
        return g

    def next_hdf5(self, layer, data, dname, nxt, chunk):
        """After pretraining one layer, move
        data to new temporary hdf5 store.
        """
        n = data.shape[0]
        d = layer.shape[1]
        tmp = nxt.create_dataset(name=dname, shape=(n, d), dtype=data.dtype)
        for i in xrange(0, n, chunk):
            tmp[i:i+chunk] = layer._fward(data[i:i+chunk])
        return tmp
        
    def _fward(self, data):
        for layer in self:
            data = layer._fward(data)
        return loss_table[self._score](data, targets=None, predict=True)
