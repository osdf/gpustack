"""

"""


import numpy as np
from itertools import izip, cycle, repeat
import json


from gnumpy import garray
from gnumpy import max as gmax
from gnumpy import sum as gsum
from gnumpy import newaxis as gnewaxis
from gnumpy import exp as gexp
from gnumpy import log as glog
import chopmunk as munk


def _cycle(data, btsz):
    """
    """
    bgn = cycle(xrange(0, data.shape[0]-btsz+1, btsz))
    end = cycle(xrange(btsz, data.shape[0]+1, btsz))
    return bgn, end


def cycle_inpt(inputs, btsz, **kwargs):
    """
    """
    bgn, end = _cycle(inputs, btsz)
    for idx, idx_p1 in izip(bgn, end):
        yield garray(inputs[idx:idx_p1].copy())


def cycle_noisy_inpt(inputs, btsz, noise, **kwargs):
    """
    """
    bgn, end = _cycle(inputs, btsz)
    for idx, idx_p1 in izip(bgn, end):
        _inputs = inputs[idx:idx_p1]
        noisify = np.random.rand(*_inputs.shape) > noise
        noisify = noisify * _inputs
        yield garray(noisify)


def cycle_trgt(targets, btsz, **kwargs):
    """
    """
    bgn, end = _cycle(targets, btsz)
    for idx, idx_p1 in izip(bgn, end):
        yield garray(targets[idx:idx_p1])


def jump(frm, to, when):
    i=0
    while True:
        if i >= when:
            yield to
        else:
            yield frm
            i = i+1


def range_inpt(inputs, btsz, **kwargs):
    return lambda idx: garray(inputs[idx:idx+btsz])


def range_trgt(targets, btsz, **kwargs):
    return lambda idx: garray(targets[idx:idx+btsz])


def range_noisy_inpt(inputs, btsz, noise, **kwargs):
    def noisify(idx):
        _inputs = inputs[idx:idx+btsz]
        noisify = np.random.rand(*_inputs.shape) > noise
        noisify = noisify * _inputs
        return garray(noisify)
    return noisify


external_iargs = {
    cycle_inpt: {"inputs": "inputs"}
    ,cycle_noisy_inpt: {"inputs": "inputs", "noise": "noise"}
    ,cycle_trgt: {"targets": "targets"}
}


finite_arg = {
    cycle_inpt: range_inpt
    ,cycle_noisy_inpt: range_noisy_inpt
    ,cycle_trgt: range_trgt
}


def logsumexp(array, axis=0):
    """
    Compute log of (sum of exps) 
    along _axis_ in _array_ in a 
    stable way.
    """
    axis_max = gmax(array, axis)[:, gnewaxis]
    return axis_max + glog(gsum(gexp(array-axis_max), axis))[:, gnewaxis]


def _logsumexp(array, axis=0):
    """
    """
    axis_max = np.max(array, axis)[:, np.newaxis]
    return axis_max + np.log(np.sum(np.exp(array-axis_max), axis))[:, np.newaxis]


def prepare_opt(opt_schedule, schedule, train, valid):
    # iargs, a generator passed to climin optimizer,
    # is build out of generators on the fly -- needs to know what
    # parameters those generators must be called with.
    
    otype = opt_schedule["type"]
    opt = otype.__new__(otype)
    opt_keys = opt_schedule.keys()

    opt_schedule["inputs"] = train[0]
    opt_schedule["targets"] = train[1]

    iargs=[]
    for arg in opt_schedule["iargs"]:
        needed_args = external_iargs[arg]
        for n in needed_args:
            # get only arguments that are not yet available
            if n not in opt_schedule:
                opt_schedule[n] = schedule[needed_args[n]]
        iargs.append(arg(**opt_schedule))
    iargs = izip(*iargs)
    
    ikwargs = repeat({})
    
    opt_schedule["train"] = train
    opt_schedule["valid"] = valid
    if "eval" not in opt_schedule:
        opt_schedule["eval"] = schedule["eval"]

    evals = eval_opt(opt_schedule)

    for arg in opt_schedule["iargs"]:
        needed_args = external_iargs[arg]
        for n in needed_args:
            if n in opt_schedule and n not in opt_keys:
                del opt_schedule[n]
    return opt, iargs, ikwargs, evals


def eval_opt(schedule):
    btsz = schedule["btsz"]
    score = schedule["f"]
    evals = {}

    for e in schedule["eval"]:
        args = []
        schedule["inputs"] = schedule[e][0]
        schedule["targets"] = schedule[e][1]
        for arg in schedule["iargs"]:
            args.append(finite_arg[arg](**schedule))
        inputs = schedule["inputs"]

        def loss(wrt, inputs=inputs, args=args):
            acc = 0
            N = inputs.shape[0]
            for idx in xrange(0, N - btsz + 1, btsz):
                acc += score(wrt, *[arg(idx) for arg in args])
            return acc/N

        evals[e] = loss
    return evals


def replace_gnumpy_data(item):
    if isinstance(item, dict):
        item = dict((k, replace_gnumpy_data(item[k])) for k in item)
    elif isinstance(item, list):
        item = [replace_gnumpy_data(i) for i in item]
    elif isinstance(item, tuple):
        item = tuple(replace_gnumpy_data(i) for i in item)
    elif isinstance(item, garray):
        if item.size > 1:
            item = item.abs().mean()
    return item


def load_params(fname):
    d = dict()
    with open(fname) as f:
        for line in f:
            tmp = json.loads(line)
            tmp["params"] = np.asarray(tmp["params"], dtype=np.float32)
            d[tmp["layer"]] = tmp
    return d


def log_queue(log_to):
    # standard logfile
    jlog = munk.file_sink(log_to+".log")
    jlog = munk.jsonify(jlog)
    jlog = munk.timify(jlog, tag="timestamp")
    jlog = munk.exclude(jlog, "params")

    # parameter logfile
    paraml = munk.file_sink(log_to+".params")
    paraml = munk.jsonify(paraml)
    paraml = munk.timify(paraml, tag="timestamp")
    paraml = munk.include(paraml, "params")

    jplog = munk.broadcast(*[jlog, paraml])

    # finally a pretty printer for some immediate feedback
    pp = munk.timify(munk.prettyprint_sink())
    pp = munk.dontkeep(pp, "tags")
    pp = munk.include_tags_only(pp, "pretty")

    jplog = munk.exclude_tags(jplog, "pretty")

    log = munk.broadcast(*[jplog, pp])
    return log
