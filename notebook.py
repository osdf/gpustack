"""
Take care of managing experiments: storing, retrieving.
"""


import sys
sys.path.append("../")


import types
import hashlib, shutil
import cPickle
import os, glob
from os.path import join, exists
from time import strftime
import json


def func_str(func):
    """
    convert a function into a string.
    Standard function string comes with
    address of function, avoid that,
    for hashing reasons. Also look at
    bytecode of function. Possibly not
    optimal solution.
    """
    fct_hash = str(hash(str(func.__code__.co_code)))
    return "<" + str(func).split()[1] + "-" + fct_hash + ">"


def gen_str(gen):
    """
    """
    g_hash = str(hash(str(gen.gi_code.co_code)))
    hsh = "<" + str(gen).split()[2] + "-" + g_hash + ">"
    return hsh


def tuple_str(tup):
    """
    Convert a tuple to string.
    """
    stringed = "("
    for elems in tup:
        stringed += atom_str(elems) + ","
    stringed += ")"
    return stringed


def dict_str(dic):
    """
    Convert a dictionary to a hashable
    string. Avoid arbitrarines by sorting
    keys.
    """
    stringed = "{"
    keys = dic.keys()
    keys.sort()
    for k in keys:
        stringed += str(k) + ": " + atom_str(dic[k]) + "\n"
    stringed += "}"
    return stringed


def atom_str(atom):
    """
    Given an element, make a nice string out of it.
    """
    if type(atom) is types.FunctionType:
        stringed = func_str(atom)
    elif type(atom) is types.GeneratorType:
        stringed = gen_str(atom)
    elif type(atom) is dict:
        stringed = dict_str(atom)
    elif type(atom) is tuple:
        stringed = tuple_str(atom)
    else:
        stringed = str(atom)
    return stringed


def prepare(schedule, depot="depot") :
    """
    Prepare things for notebook.

    - generate a directory out of a schedule-hash
    - copy over labfile and config file
    - dump (cPickle) schedule
    """
    # _now_ is the time right now
    now = strftime("%Y-%m-%d-%H:%M:%S")
    # hash the lab file: if this file is changed,
    # consider it a different experiment
    # "__lab__" key was put by dispatch.py
    lab = open(schedule["__lab__"])
    schedule["__lab__#"] = hashlib.sha1(lab.read()).hexdigest()
    # Note: Don't hash the config file. A config file
    # can define several single experiments. A specific 
    # experiment may come from two different config files.
    # Instead, hash the schedule.
    to_hash = dict_str(schedule)
    folder = hashlib.sha1(to_hash).hexdigest()[0:20]
    path = join(depot, folder)
    if not exists(path):
        os.makedirs(path)
        schedfile = join(path, now) + ".schedule"
        with open(schedfile, "w") as f:
            f.write(json.dumps(to_hash))
    else:
        print "[NOTEBOOK:prepare] It seems, you are _re_running an experiment."
        print "[NOTEBOOK:prepare] Found already this hash:", folder
        print "[NOTEBOOK:prepare] _schedule_ will not be jsonified.\n"
    # get config and lab files
    path = join(path, now) 
    shutil.copy(schedule["__lab__"], path + ".lab")
    # even though we didn't hash the config file, it is good
    # to know, which config's gave rise to this experiment.
    shutil.copy(schedule["__config__"], path + ".config")
    print "[NOTEBOOK:prepare] Working on %s\n"%path
    return path


def parameters(path, prms):
    """
    Dump parameters in path.
    """
    prm = open(path + ".prms", "wb")
    cPickle.dump(prms, prm)
    prm.close()
    #
    print "[NOTEBOOK:parameters] Saved parameters in", path


def finalize(path, result):
    """
    Dump results in path,
    build via hashing in prepare.
    """
    #
    res = open(path + ".res", "wb")
    cPickle.dump(result, res)
    res.close()
    #
    print "[NOTEBOOK:finalize] Saved results in", path


def single(spec, location):
    """
    Retrieve a specific experiment
    at _location_. _spec_ allows to
    define specifications that must
    be fullfilled. This is mostly
    interesting for the general retrieve
    case.
    """
    # one scheulde per directory
    sf = open(glob.glob(location+"/*.schedule")[0])
    sched = cPickle.load(sf)
    sf.close()
    #
    if not satisfy(spec, sched, key="schedule"):
        return None
    results = glob.glob(dirname+"/*.res")
    if (len(results) == 0):
        print "[NOTEBOOK:retrieve] No results available!"
        print "[NOTEBOOK:retrieve] Check", dirname, "\n"
        return None 
    # collect results/parameters fullfilling 'spec'
    rdic = {}
    pdic = {}
    for r in results:
        # open result file
        rf = open(r, "rb")
        res = cPickle.load(rf)
        rf.close()
        # filter res dictionary
        if not satisfy(spec, res, key="result"):
            continue
        # get parameters -- filename ends with .prms
        tag = r.split('.')[0]
        p = tag + ".prms"
        pf = open(p, "rb")
        prms = cPickle.load(pf)
        pf.close()
        if not satisfy(spec, prms, key="parameters"):
            continue
        # results/parameters ok for 'spec'
        key = tag.split('/')[-1]
        rdic[key] = res
        pdic[key] = prms
    if ((len(rdic) == 0) or (len(pdic) == 0)):
        return None
    return {"schedule":sched, "results": res, "parameters":prms}


def retrieve(spec={}, depot="depot/"):
    """
    Retrieve collected experiments from _depot_.
    Check if _spec_ is satisfied by the
    experiments.

    _spec_ is a dictionary, that lists several
    constraints (e.g. schedule should have only
    100 epochs) for keys "schedule", "result" and
    "parameters". _depot_ needs a trailing "/".
    """
    db = {}
    for d in os.listdir(depot):
        dirname = depot + d
        if os.path.isdir(dirname):
            tmp = single(spec, location)
            if tmp is not None:
                db[d] = tmp
    # hint on empty case
    if (len(db) == 0):
        print "[NOTEBOOK:retrieve] Retrieval found nothing."
        print "[NOTEBOOK:retrieve] Check keys in _spec_ and _depot_ name, ",\
                "must end in '/'."
    return db


def here(spec=None):
    """
    Retrieve experiment of current directory.
    (Assumes, that current directory is valid
    epxeriment directory.)
    """
    location = os.getcwd()
    return single(spec, location)


def satisfy(spec, dic, key):
    """
    Check, if dic satisfies all
    constraints listed in spec[key].
    """
    # _spec_ does not have _key_:
    # no constraints -> we are ok.
    if spec.has_key(key):
        sat = spec[key]
        for k in sat:
            if dic.has_key(k):
                if sat[k] != dic[k]:
                    return False
            else:
                return False
    # we are fine.
    return True


def overview(keys, spec={}, depot="depot/"):
    """
    Give an overview over all experiments
    in _depot_ that fullfill constraints
    in _spec_. Return only those values that
    are specified in _keys_.

    The assumption is hereby that every experiment
    in _depot_ is described as a dictionary.
    In _keys_ one specifies those keys of the
    experiment that should be returned as a list,
    or, if experiments are more nested, as an
    arbitrary dictionary.
    """
    db = retrieve(spec, depot)
    view = {}
    for h, exp in db.items():
        empty = True
        key_dic = dict()
        for key in keys:
            if len(keys[key]) == 0:
                visibles = exp[key].keys()
            else:
                visibles = keys[key]
            key_dic[key] = dict((k, v) for k, v in exp[key].items() if k in visibles)
            if len(key_dic[key]) > 0:
                empty = False
        if empty:
            print "[NOTEBOOK:overview] Warning: Check _keys_, empty view for", h
        else:
            view[h] = key_dic
    return view
