"""

"""


import numpy as np
import random
from itertools import izip, cycle, repeat
import json


from gnumpy import garray
from gnumpy import max as gmax
from gnumpy import sum as gsum
from gnumpy import newaxis as gnewaxis
from gnumpy import exp as gexp
from gnumpy import log as glog
import chopmunk as munk
import climin.util


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
        yield garray(inputs[idx:idx_p1])


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


def cycle_pairs(pairs, btsz, **kwargs):
    """
    """
    p0, p1 = pairs[0], pairs[1]
    bg, end = _cycle(p0, btsz)
    for idx, idx_p1 in izip(bg, end):
        yield (garray(p0[idx:idx_p1]), garray(p1[idx:idx_p1]))


def jump(frm, to, when):
    i = 0
    while True:
        if i >= when:
            yield to
        else:
            yield frm
            i = i+1


def lin_inc(frm, to, step, end):
    i = 0
    diff = to - frm
    delta = end/(1.0*step)
    inc = diff/delta
    # minus inc handels divmod/i=0 case.
    strt = frm - inc
    while True:
        if i >= end:
            yield to
        else:
            d, r = divmod(i, step)
            if r == 0:
                strt += inc
            yield strt
            i = i + 1


def const(const):
    while True:
        yield const


def two_step(step_one, step_two):
    for s1, s2 in izip(step_one, step_two):
        yield (s1, s2)


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


def prepare_opt(opt_schedule, wrt, schedule, train, valid):
    # iargs, a generator passed to climin optimizer,
    # is build out of generators on the fly -- needs to know what
    # parameters those generators must be called with.
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

    opt_keys = opt_schedule.keys()
    for arg in opt_schedule["iargs"]:
        needed_args = external_iargs[arg]
        for n in needed_args:
            if n in opt_schedule and n not in opt_keys:
                del opt_schedule[n]
    # get optimizer
    opt = opt_schedule["type"]
    opt_schedule["args"] = izip(iargs, ikwargs)
    opt = climin.util.optimizer(opt, wrt, **opt_schedule)
    return opt, evals


def eval_opt(schedule):
    btsz = schedule["btsz"]
    if "eval_score" in schedule:
        score = schedule["eval_score"]
    else:
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
            return acc

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


def log_queue(log_to=None):
    if log_to:
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
    else:
        pp = munk.timify(munk.prettyprint_sink())
        pp = munk.dontkeep(pp, "tags")
        log = munk.include_tags_only(pp, "pretty")
    return log


def reload(depot, folder, tag, layer):
    """
    """
    import notebook as nb
    model, schedule = nb.reload(depot, folder, tag, layer)

    log = munk.prettyprint_sink()
    log = munk.dontkeep(log, "tags")
    log = munk.include_tags_only(log, "pretty")

    schedule['logging'] = log

    lab = schedule['__lab__']
    lab = __import__(lab.split('.')[0])
    lab.no_training(model, schedule)


def init_SI(shape, sparsity):
    """
    Produce sparsely initialized weight matrix
    as described by Martens, 2010.

    Note: shape is supposed to be visible x hiddens.
    The following code produces first a hiddens x visible.
    """
    tmp = np.zeros((shape[1], shape[0]))
    for i in tmp:
        i[random.sample(xrange(shape[0]), sparsity)] = np.random.randn(sparsity)
    return tmp.T


def binomial(width):
    filt = np.array([0.5, 0.5])
    for i in xrange(width-2):
        filt = np.convolve(filt, [0.5, 0.5])
    return filt


def mask(factors, stride, size):
    fsqr = int(np.sqrt(factors))
    hsqr = int(fsqr/stride)
    conv = np.zeros((factors, hsqr*hsqr), dtype=np.float32)
    msk = np.zeros((factors, hsqr*hsqr), dtype=np.float32)
    _s = size/2
    print "Mask size:", msk.shape
    col = np.zeros((1, fsqr))
    col[0, 0:size] = binomial(size) 
    row = np.zeros((1, fsqr))
    row[0, 0:size] = binomial(size) 
    for j in xrange(0, fsqr, stride):
        for i in xrange(0, fsqr, stride):
            _row = np.roll(row, j-_s)
            _col = np.roll(col, i-_s)
            idx = (j*hsqr + i)/stride
            conv[:, idx] = np.dot(_col.T, _row).ravel()
            msk[:, idx] = conv[:, idx] > 0 
    return msk, conv
